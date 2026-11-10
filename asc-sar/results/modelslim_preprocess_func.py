# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os

import numpy as np

from modelslim import logger


MAX_PIXEL_VALUE = 255.0
FLOAT32 = "float32"


def check_h_w(height, width):
    if height is None or width is None:
        raise ValueError("The height and width of the model input cannot be None.")


def preprocess_func_imagenet(data_path, height=224, width=224, batch_size=1):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    return: list of preprocess image data
    """
    import cv2
    
    check_h_w(height, width)
    image_list = os.listdir(data_path)
    data_list = []
    for idx, image_name in enumerate(image_list):
        image_filepath = os.path.join(data_path, image_name)
        img = cv2.imread(image_filepath)
        if img is None:
            logger.warning("File %s is invalid.", image_filepath)
            continue
        img_data = cv2.resize(img, (width, height))
        img_data = img_data[:, :, ::-1].transpose(2, 0, 1)

        # normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype(FLOAT32)
        for j in range(img_data.shape[0]):
            norm_img_data[j, :, :] = (img_data[j, :, :] / MAX_PIXEL_VALUE - mean_vec[j]) / stddev_vec[j]

        data_list.append(norm_img_data)
    return get_all_batch(data_list, batch_size)


def preprocess_func_coco(data_path, height=320, width=320, batch_size=1):
    import cv2

    check_h_w(height, width)
    file_list = os.listdir(data_path)
    data_list = []
    for file_name in file_list:
        image_filepath = os.path.join(data_path, file_name)
        img = cv2.imread(image_filepath)
        if img is None:
            logger.warning("File %s is invalid.", image_filepath)
            continue
        img_data = cv2.resize(img, (width, height))
        img_data = img_data[:, :, ::-1].transpose(2, 0, 1).astype(FLOAT32)
        img_data /= MAX_PIXEL_VALUE
        data_list.append(img_data)
    return get_all_batch(data_list, batch_size)


def get_all_batch(data_list, batch_size):
    all_batch_data = []
    data_size = len(data_list)
    if data_size < batch_size:
        raise ValueError("The number of calibrated images is smaller than the batch size, "
                         "please add some pictures.")
    for batch_start in range(0, data_size // batch_size * batch_size, batch_size):
        per_batch_data = data_list[batch_start: batch_start + batch_size]
        per_batch_data = np.array(per_batch_data)
        all_batch_data.append(per_batch_data)
    return all_batch_data
