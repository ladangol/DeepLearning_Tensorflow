from util import get_path, get_index_by_image_name

import random
import numpy as np
from numpy import save
import os
from os import listdir
import cv2

import keras
from sklearn.model_selection import train_test_split
small_data_set_size = 20

def split_and_save(in_config, data,labels):
    print("Preprocessing data!")
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42, shuffle=True)

    trainY = keras.utils.to_categorical(trainY, in_config.num_classes)
    testY = keras.utils.to_categorical(testY,  in_config.num_classes)

    # save the reshaped photos
    trainX_path = get_path(in_config.data_path_root, in_config.trainX_path)
    save(trainX_path, trainX)

    trainY_path = get_path(in_config.data_path_root, in_config.trainY_path)
    save(trainY_path, trainY)

    testX_path = get_path(in_config.data_path_root, in_config.testX_path)
    save(testX_path, testX)

    testY_path = get_path(in_config.data_path_root, in_config.testY_path)
    save(testY_path, testY)

def prepare_data(in_data_dir, in_config):
    print("Start preparing data!")
    image_paths = []

    # enumerate files in the directory
    for file in listdir(in_data_dir):
        image_path = get_path(in_data_dir, file)  # create path to dogs and cats
        image_paths.append(image_path)

    random.seed(42)
    random.shuffle(image_paths)
    data, labels = list(), list()

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (in_config.image_size, in_config.image_size))
        data.append(image)

        # determine class
        image_name = os.path.basename(image_path)
        index = get_index_by_image_name(in_config, image_name)
        if index == -1:
            print("unrecognizede label for image:" + image_path)
            exit(-1)

        output = float(index)
        labels.append(output)
        if(len(labels)==small_data_set_size):
            break

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    split_and_save(in_config, data, labels)

    print("Data preparation completed!")
    return data, labels

