import argparse

import pandas as pd
from generator import *
from model import segnet
import cv2
import numpy as np
from keras.callbacks import CSVLogger


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')#, description="SegNet LIP dataset")
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
    parser.add_argument("--optimizer", default="adadelta", type=str, help="oprimizer")
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

    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])
    csv_logger = CSVLogger('training.log')
    model.fit_generator(
        train_gen,
        steps_per_epoch=args.epoch_steps,
        epochs=args.n_epochs,
        validation_data=val_gen,
        validation_steps=args.val_steps,
        callbacks=[csv_logger],
    )

    model.save_weights(args.save_dir + str(args.n_epochs) + ".hdf5")
    print("sava weight done..")
    #print(args.save_dir + str(args.n_epochs) + ".hdf5")
    #file_path = args.save_dir + str(args.n_epochs) + ".hdf5"
    #model.load_weights(file_path)

    save_path = "results/"
    count = 0
    for i,(image,mask) in enumerate(test_gen):
      pred_mask = model.predict(image)
      image = rgb2gray(image).squeeze()
      mask = mask.argmax(axis=2).reshape(args.input_shape[0],args.input_shape[1])
      pred_mask = pred_mask.argmax(axis=2).reshape(args.input_shape[0],args.input_shape[1])
      print(image.shape)
      print(mask.shape)
      print(pred_mask.shape)
      print(np.max(image),np.min(image))
      print(np.max(mask),np.min(mask))
      print(np.max(pred_mask),np.min(pred_mask))

      sep_line = np.ones((args.input_shape[0], 10)) * 255
      all_images = [image * 255, sep_line, mask * 255, sep_line, pred_mask * 255]
      cv2.imwrite(f"{save_path}/{i}.png", np.concatenate(all_images, axis=1))

      count += 1
      if count == len(val_list):
        break

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == "__main__":
    args = argparser()
    main(args)
