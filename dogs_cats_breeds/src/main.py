import config
from util import get_path, plot_confusion_matrix, get_categories
from data_prepration import prepare_data

from train import train, predict


def print_main_menu():
    print('press d for data_preparation: ')
    print('press t for train: ')
    print('press p for prediction: ')
    print('press e for exit: ')

def main():
    print_main_menu()
    action = input()
    if action == 'd':
        # define location of dataset
        train_data_path = get_path(config.data_path_root, 'train')
        prepare_data(train_data_path, config)
    elif action == 'e':
        return
    elif action == 't':
        train(config)

    elif action == 'p':
        test_model_path = get_path(config.model_path_root, 'dog_cat_breed.model')
        test_data_path = get_path(config.data_path_root, 'test')
        predict(test_data_path, test_model_path, config)


main()