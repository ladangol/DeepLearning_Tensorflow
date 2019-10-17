from util import get_path, get_index_by_image_name

import random
from os import listdir
import numpy as np
import os
import cv2
from numpy import save

def prepare_data(in_data_dir, in_config):
    print("Start preparing data!")
    in_output_data_path = get_path(in_config.data_path_root, in_config.data_name)
    in_output_labels_path = get_path(in_config.data_path_root, in_config.labels_name)

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

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # save the reshaped photos
    save(in_output_data_path, data)
    save(in_output_labels_path, labels)

    print("Data preparation completed!")
    return data, labels

