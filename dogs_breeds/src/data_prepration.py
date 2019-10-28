from util import get_path, write_labels
from zipfile import ZipFile
import os
import csv
import cv2
import numpy as np
from numpy import save

max_call_use = 500

def add_to_list(in_config, in_row, inout_data, inout_labels, inout_dict_categories, inout_breed_count):
    image_name = in_row[0]
    label = in_row[1]

    data_path = get_path(in_config.data_path_root, 'train')
    image_path = get_path(data_path, image_name) + ".jpg"
    if os.path.exists(get_path(in_config.data_path_root, 'train')):
        if not label in inout_dict_categories:
            if len(inout_dict_categories)== max_call_use:
                return
            inout_dict_categories[label] = len(inout_dict_categories)
            inout_breed_count[label] = 0

        image = cv2.imread(image_path)
        image = cv2.resize(image, (in_config.image_size, in_config.image_size))
        inout_data.append(image)

        # determine class
        index = inout_dict_categories[label]
        inout_labels.append(index)

        count = inout_breed_count[label]
        inout_breed_count[label] = count + 1

def prepare_data(in_config):
    print("Start preparing data!")

    if not os.path.exists(get_path(in_config.data_path_root,'test')):
        with ZipFile(get_path(in_config.data_path_root,'test.zip'), 'r') as zipObj:
           # Extract all the contents of zip file in different directory
           zipObj.extractall(in_config.data_path_root)

    if not os.path.exists(get_path(in_config.data_path_root,'train')):
        with ZipFile(get_path(in_config.data_path_root,'train.zip'), 'r') as zipObj:
           # Extract all the contents of zip file in different directory
           zipObj.extractall(in_config.data_path_root)

    data, labels = list(), list()

    label_path = get_path(in_config.data_path_root,'labels.csv')
    with open(label_path) as csv_file:
        labels_raw = csv.reader(csv_file, delimiter=',')
        next(labels_raw, None)

        dict_categories = {}
        breed_count = {}
        for row in labels_raw:
            add_to_list(in_config, row, data, labels, dict_categories, breed_count)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # save the reshaped photos
    output_data_path = get_path(in_config.data_path_root, in_config.data_name)
    output_labels_path = get_path(in_config.data_path_root, in_config.labels_name)
    save(output_data_path, data)
    save(output_labels_path, labels)

    inv_categories = {v: k for k, v in dict_categories.items()}
    lables_path = get_path(in_config.data_path_root, in_config.lable_index_map)
    write_labels(lables_path, inv_categories)

    breeds_distribution_file = get_path(in_config.data_path_root, "breeds_distribution.json")
    write_labels(breeds_distribution_file, breed_count)

    print("Data preparation completed!")
    return data, labels
