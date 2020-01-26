from util import get_path
from util import get_index_by_image_name

import random
import numpy as np
from numpy import save
import os
from os import listdir
import cv2
import keras
from sklearn.model_selection import train_test_split

small_data_set_size = 20

def split_and_save(config, data,labels):
    print("Pre-processing data!")
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels,
                                                          test_size=0.25,
                                                          stratify=labels,
                                                          random_state=42,
                                                          shuffle=True)

    #todo:remove it from here, add it to when you do training since we might need it for embbeding
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    # save the reshaped photos
    x_train_path = get_path(config.data_path_root, config.trainX_path)
    save(x_train_path, x_train)

    y_train_path = get_path(config.data_path_root, config.trainY_path)
    save(y_train_path, y_train)

    x_test_path = get_path(config.data_path_root, config.testX_path)
    save(x_test_path, x_test)

    y_test_path = get_path(config.data_path_root, config.testY_path)
    save(y_test_path, y_test)

def prepare_data(data_dir, config):
    print("Start preparing data!")
    image_paths = []

    # enumerate files in the directory
    for file in listdir(data_dir):
        image_path = get_path(data_dir, file)  # create path to dogs and cats
        image_paths.append(image_path)

    random.seed(42)
    random.shuffle(image_paths)
    data, labels = list(), list()

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (config.image_size, config.image_size))
        data.append(image)

        # determine class
        image_name = os.path.basename(image_path)
        index = get_index_by_image_name(config, image_name)
        if index == -1:
            print("unrecognizede label for image:" + image_path)
            exit(-1)

        output = float(index)
        labels.append(output)

        # if(len(labels)==small_data_set_size):
        #     break

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    split_and_save(config, data, labels)

    print("Data preparation completed!")
    return data, labels

