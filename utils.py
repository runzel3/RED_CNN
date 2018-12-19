import os
import cv2
import scipy
import numpy as np
from scipy import ndimage

data_dir = './data'

PATCH_SIZE = 128

def get_image_path(is_test, s, num):
    assert (s == 128 or s == 64)
    # path = os.path.join('/Users/hthieu/PycharmProjects/CS446_Final_Project', "xray_images/")
    path = os.path.join(os.getcwd(), "xray_images")
    img_dir = ""
    image_name = ""
    if is_test:
        img_dir += 'test_images_'
        image_name += 'test_'
    else:
        img_dir += 'train_images_'
        image_name += 'train_'
    if s == 64:
        img_dir += '64x64'
    elif s == 128:
        img_dir += '128x128'

    path = os.path.join(path, img_dir)
    num_str = format(num, "05")
    image_name += num_str + ".png"
    path = os.path.join(path, image_name)
    return path


def imread(path):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32')

def scale_image(img, factor=2.0):
    return ndimage.interpolation.zoom(img, factor, prefilter=False)

def load_data():
    print("Start loading input...")
    input_file = os.path.join(data_dir, 'inputs.npy')
    label_file = os.path.join(data_dir, 'labels.npy')
    input_ = np.load(input_file)
    print("Finished loading input...")
    label_ = np.load(label_file)
    print("Finished loading label...")
    return input_, label_

def load_data_directly():
    print("Start loading input...")
    start_idx = 4000
    end_idx = 20000
    total_num_patch = end_idx - start_idx
    print("Total number of patches " + str(total_num_patch))
    data = np.zeros((total_num_patch, PATCH_SIZE, PATCH_SIZE, 1))
    labels = np.zeros((total_num_patch, PATCH_SIZE, PATCH_SIZE, 1))
    cur_idx = 0
    for i in range(start_idx, end_idx):
        if i % 500 == 0:
            print("Processing image number " + str(i) + "...")
        noisy_img = imread(get_image_path(False, 64, i))  # Image size 64x64
        noisy_img = scale_image(noisy_img, 2.0)  # Image size 128x128
        noisy_img /= 255.0
        clean_img = imread(get_image_path(False, 128, i))  # Image size 128x128
        clean_img /= 255.0
        im_h, im_w = noisy_img.shape
        noisy_img = noisy_img.reshape([im_h, im_w, 1])
        clean_img = clean_img.reshape([im_h, im_w, 1])
        data[cur_idx] = noisy_img
        labels[cur_idx] = clean_img
        cur_idx += 1
    print("Total number of patches=", cur_idx+1)
    return data, labels
