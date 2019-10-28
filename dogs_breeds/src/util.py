import os
import json

def get_path(in_root, in_folder):
    return os.path.join(in_root, in_folder)

def read_labels(in_config):
    lables_path = get_path(in_config.data_path_root, in_config.lable_index_map)
    with open(lables_path) as handle:
        in_config.dict_categories = json.load(handle)
    in_config.num_classes = len(in_config.dict_categories)

def write_labels(in_file_name, in_dict_categories):
    with open(in_file_name, 'w') as fp:
        json.dump(in_dict_categories, fp, sort_keys=True, indent=4)



