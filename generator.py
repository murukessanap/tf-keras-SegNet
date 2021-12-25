import cv2
import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            img_path = img_dir + str(lists.iloc[i, 0]) + ".jpg"
            #print(img_path)
            
            #original_img = cv2.imread(img_path)[:, :, ::-1]
            #resized_img = cv2.resize(original_img, (dims[0],dims[1]))
            
            original_img = Image.open(img_path).convert("RGB")
            resized_img = original_img.resize((dims[0],dims[1]))
            resized_img = np.asarray(resized_img) / 255.0
            resized_img = np.expand_dims(resized_img,-1)
            
            #array_img = img_to_array(resized_img) / 255
            imgs.append(resized_img) 
            # masks
            mask_path = mask_dir + str(lists.iloc[i, 0]) + "_Al.ome.tiff"
            #print(mask_path)
            #original_mask = cv2.imread(mask_dir + str(lists.iloc[i, 0]) + ".ome.tiff")
            
            #original_mask = cv2.imread(mask_path,0)
            #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            
            original_mask = Image.open(mask_path).convert("L") 
            resized_mask = original_mask.resize((dims[0], dims[1]))
            resized_mask = np.asarray(resized_mask)
            resized_mask = np.expand_dims(resized_mask,-1)
            
            #array_mask = category_label(resized_mask, (dims[0], dims[1]), n_labels)
            array_mask = resized_mask
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels

def data_gen_test(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    count = 0
    while True:
        ix = np.arange(len(lists))[count*batch_size:(count+1)*batch_size]
        imgs = []
        labels = []
        img_paths = []
        for i in ix:
            # images
            img_path = img_dir + str(lists.iloc[i, 0]) + ".jpg"
            img_paths.append(img_path)
            #print(img_path)
            
            #original_img = cv2.imread(img_path)[:, :, ::-1]
            #resized_img = cv2.resize(original_img, (dims[0],dims[1]))
            
            original_img = Image.open(img_path).convert("RGB")
            resized_img = original_img.resize((dims[0],dims[1]))
            resized_img = np.asarray(resized_img) / 255.0
            resized_img = np.expand_dims(resized_img,-1)
            
            #array_img = img_to_array(resized_img) / 255
            imgs.append(resized_img)
            # masks
            mask_path = mask_dir + str(lists.iloc[i, 0]) + "_Al.ome.tiff"
            
            #original_mask = cv2.imread(mask_path,0)
            #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            
            original_mask = Image.open(mask_path).convert("L") 
            resized_mask = original_mask.resize((dims[0], dims[1]))
            resized_mask = np.asarray(resized_mask)
            resized_mask = np.expand_dims(resized_mask,-1)
            
            #array_mask = category_label(resized_mask, (dims[0], dims[1]), n_labels)
            array_mask = resized_mask
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        count += 1
        yield imgs, labels, img_paths        
