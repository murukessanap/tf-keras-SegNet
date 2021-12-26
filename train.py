import argparse

import pandas as pd
from generator import *
from model import segnet
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger,ModelCheckpoint
import keras.backend as K
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
import keras


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')#, description="SegNet LIP dataset")
    parser.add_argument("--lr", default="0.001", help="learning rate")
    parser.add_argument("--save_dir", help="output directory")
    parser.add_argument("--train_list", help="train list path")
    parser.add_argument("--trainimg_dir", help="train image dir path")
    parser.add_argument("--trainmsk_dir", help="train mask dir path")
    parser.add_argument("--val_list", help="val list path")
    parser.add_argument("--valimg_dir", help="val image dir path")
    parser.add_argument("--valmsk_dir", help="val mask dir path")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--n_epochs", default=10, type=int, help="number of epoch")
    parser.add_argument(
        "--epoch_steps", default=100, type=int, help="number of epoch step"
    )
    parser.add_argument(
        "--val_steps", default=10, type=int, help="number of valdation step"
    )
    parser.add_argument("--n_labels", default=20, type=int, help="Number of label")
    parser.add_argument(
        "--input_shape", default=(256, 256, 3), help="Input images shape"
    )
    parser.add_argument("--kernel", default=3, type=int, help="Kernel size")
    parser.add_argument(
        "--pool_size", default=(2, 2), help="pooling and unpooling size"
    )
    parser.add_argument(
        "--output_mode", default="softmax", type=str, help="output activation"
    )
    parser.add_argument(
        "--loss", default="categorical_crossentropy", type=str, help="loss function"
    )
    parser.add_argument("--optimizer", default="adadelta", type=str, help="optimizer")
    parser.add_argument("--model_path", default="/content/segnet_model.h5", type=str, help="save best model")
    
    args = parser.parse_args()

    return args


def main(args):
    # set the necessary list
    train_list = pd.read_csv(args.train_list, header=None, dtype=str)
    val_list = pd.read_csv(args.val_list, header=None, dtype=str)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    train_gen = data_gen_small(
        trainimg_dir,
        trainmsk_dir,
        train_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )
    val_gen = data_gen_small(
        valimg_dir,
        valmsk_dir,
        val_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )
    test_gen = data_gen_test(
        valimg_dir,
        valmsk_dir,
        val_list,
        1,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )


    model = segnet(
        args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode
    )
    print(model.summary())

    optimizer = keras.optimizers.Adam(lr=float(args.lr))
    model.compile(loss=args.loss, optimizer=optimizer, metrics=["accuracy"])
    csv_logger = CSVLogger('training.log')
    #checkpoint = ModelCheckpoint(args.model_path, verbose=1, save_best_only=True, monitor='val_acc', mode='max')
    model.fit_generator(
        train_gen,
        steps_per_epoch=args.epoch_steps,
        epochs=args.n_epochs,
        validation_data=val_gen,
        validation_steps=args.val_steps,
        callbacks=[csv_logger]  #, checkpoint],
    )

    model.save_weights(args.save_dir + str(args.n_epochs) + ".hdf5")
    print("sava weight done..")
    #print(args.save_dir + str(args.n_epochs) + ".hdf5")
    #file_path = args.save_dir + str(args.n_epochs) + ".hdf5"
    #model.load_weights(args.model_path)

    save_path = "results/"
    count = 0
    Dice = []
    IOU = []
    for i,(image,mask,path) in enumerate(test_gen):
      pred_mask = model.predict(image)
      image = rgb2gray(image).squeeze()
      mask = mask.reshape(args.input_shape[0],args.input_shape[1])
      pred_mask = (pred_mask > 0.5).astype(np.uint8)
      #pred_mask[pred_mask>=0.5] = 1
      #pred_mask[pred_mask<0.5] = 0
      pred_mask = pred_mask.reshape(args.input_shape[0],args.input_shape[1])
      Dice.append(DiceScore(mask,pred_mask))
      IOU.append(IoUScore(mask,pred_mask))
      print(image.shape)
      print(mask.shape)
      print(pred_mask.shape)
      print(np.max(image),np.min(image))
      print(np.max(mask),np.min(mask))
      print(np.max(pred_mask),np.min(pred_mask))

      sep_line = np.ones((args.input_shape[0], 10)) * 255
      all_images = [image * 255, sep_line, mask * 255, sep_line, pred_mask * 255]
      cv2.imwrite(f"{save_path}/{path[0].split('/')[-1][:-4]}.png", np.concatenate(all_images, axis=1))

      count += 1
      if count == len(val_list):
        break
    print("Average Test DICE score: ",sum(Dice)/len(Dice))
    print("Average Test IOU score: ",sum(IOU)/len(IOU))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

smooth=1e-6
def DiceScore(targets, inputs):
    
    #flatten label and prediction tensors
    inputs = tf.cast(K.flatten(inputs), tf.float32)
    targets = tf.cast(K.flatten(targets), tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(targets,0), tf.expand_dims(inputs,-1)))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return K.get_value(dice)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def IoUScore(targets, inputs):
    
    #flatten label and prediction tensors
    inputs = tf.cast(K.flatten(inputs), tf.float32)
    targets = tf.cast(K.flatten(targets), tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(targets,0), tf.expand_dims(inputs,-1)))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return K.get_value(IoU)

if __name__ == "__main__":
    args = argparser()
    main(args)
