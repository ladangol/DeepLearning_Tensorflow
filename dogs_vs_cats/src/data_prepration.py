from util import getPath

import random
from os import listdir
import numpy as np
import os
import cv2
from numpy import save

def prepare_data(in_data_dir, in_config):
    print("Start preparing data!")
    in_image_size = in_config.image_size
    in_categories = in_config.categories
    in_output_data_path = getPath(in_config.data_path_root, in_config.data_name)
    in_output_labels_path = getPath(in_config.data_path_root, in_config.labels_name)

    image_paths = []

    # enumerate files in the directory
    for file in listdir(in_data_dir):
        image_path = getPath(in_data_dir, file)  # create path to dogs and cats
        image_paths.append(image_path)

    random.seed(42)
    random.shuffle(image_paths)
    data, labels = list(), list()
    counter=0
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (in_image_size, in_image_size))
        data.append(image)

        # determine class
        output = float(in_categories.index("Dog"))
        label = os.path.basename(image_path)
        if label.lower().startswith('cat'):
            output = float(in_categories.index("Cat"))

        labels.append(output)
        counter = counter +1
        if counter == 11:
            break;

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # save the reshaped photos
    save(in_output_data_path, data)
    save(in_output_labels_path, labels)

    print("Data preparation completed!")
    return data, labels

