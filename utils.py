# ---------------------------------------------------------
# Python Utility Function Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import random
import scipy.misc
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import math
class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


def center_crop(img, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h

    h, w = img.shape[:2]
    h_start = int(round((h - crop_h) / 2.))
    w_start = int(round((w - crop_w) / 2.))
    # resize image
    img_crop = scipy.misc.imresize(img[h_start:h_start+crop_h, w_start:w_start+crop_w], [resize_h, resize_w])
    return img_crop


def imread(path, is_gray_scale=False, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def load_data(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, is_gray_scale=False):
    img = imread(path=image_path, is_gray_scale=is_gray_scale)

    if crop:
        cropped_img = center_crop(img, input_height, input_width, resize_height, resize_width)
    else:
        cropped_img = scipy.misc.imresize(img, [resize_height, resize_width])

    img_trans = transform(cropped_img)  # from [0, 255] to [-1., 1.]

    if is_gray_scale and (img_trans.ndim == 2):
        img_trans = np.expand_dims(img_trans, axis=2)

    return img_trans


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames

def all_files_under_ucsd(path, extension=None, append_path=True, sort=True):
    if append_path:
        filenames = [f for f in listdir(path) if isfile(join(path, f))]
    return filenames


def getListOf_imgFiles(path,extension=None, append_path=True,img_type='tif', sort=True):
    # create a list of file and sub directories
    # names in the given directory
    if append_path:
        listOfFile = os.listdir(path)
        allFiles = []
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(path, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOf_imgFiles(fullPath)
            else:
                if img_type in fullPath.split('.'):
                    allFiles.append(fullPath)
    return allFiles


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:  # color image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:  # gray scale image
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr


def image_shape(filename):
    img = Image.open(filename, mode="r")
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def transform(img):
    return img / 127.5 - 1.0


def inverse_transform(img):
    return (img + 1.) / 2.


def l2_distance(x,y):
    _flat_x = x.flatten()
    _flat_y = y.flatten()
    score  = np.linalg.norm(_flat_x-_flat_y)
    return score



def j_entropy(x,y):
    _flat_x = x.flatten()
    _flat_y = y.flatten()
    m = (_flat_x+_flat_y)/2.0
    xlog = np.log(_flat_x/m)
    ylog = np.log(_flat_y/m)
    xlog[np.isnan(xlog)]=0
    ylog[np.isnan(ylog)]=0
    score = np.sum(_flat_x*xlog-_flat_y*ylog)
    if math.isnan(score):
        import pdb
        pdb.set_trace()
        return 0
    else:
        return abs(score)


def aed_localization(predict,test,wsize=5,distance_metric='l2'):
    _shape = np.shape(predict)
    _wsize = _shape[1]-wsize+1
    _hsize = _shape[2]-wsize+1
    attention_map = np.zeros([_wsize,_hsize],dtype=np.float32)
    for i in range(_wsize):
        for j in range(_hsize):
                if distance_metric=='l2':
                    attention_map[i,j] = l2_distance(predict[:,i:i+wsize,j:j+wsize],test[:,i:i+wsize,j:j+wsize])
                elif distance_metric=='je':
                    attention_map[i, j] =j_entropy(predict[:,i:i + wsize, j:j + wsize],test[:,i:i+wsize,j:j+wsize])
                else:
                    return -1
    return attention_map
