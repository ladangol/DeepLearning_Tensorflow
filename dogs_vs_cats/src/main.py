import config
from util import get_path
from data_prepration import prepare_data
from factory import GenerateModel
from factory import GenerateTraner
from factory import GeneratePredicter

from numpy import load

class TrainingData:
    def __init__(self, in_config):

        print("Loading trainX data!")
        root = in_config.data_path_root
        trainX_path = get_path(root, in_config.trainX_path)
        self.trainX = load(trainX_path)

        print("Loading trainY data!")
        trainY_path = get_path(root, in_config.trainY_path)
        self.trainY = load(trainY_path)

        print("Loading testX data!")
        testX_path = get_path(root, in_config.testX_path)
        self.testX = load(testX_path)

        print("Loading testY data!")
        testY_path = get_path(root, in_config.testY_path)
        self.testY  = load(testY_path)

def print_main_menu():
    print('press d for data_preparation: ')
    print('press t for train: ')
    print('press p for prediction: ')
    print('press e for exit: ')

def print_train_menu():
    print('press t for train: ')
    print('press gs for grid search train: ')
    print('press e for exit: ')

def print_model_menu():
    print('press s for simple cnn model: ')
    print('press c for cam model: ')
    print('press e for exit: ')

def print_prediction_menu():
    print('press p for prediction: ')
    print('press c for cam prediction: ')
    print('press e for exit: ')

def main():
    print_main_menu()
    general_action = input()
    if general_action == 'e':
        return

    if general_action == 'd':
        # define location of dataset
        train_data_path = get_path(config.data_path_root, 'train')
        prepare_data(train_data_path, config)
        return

    if general_action == 't':
        print_train_menu()
        train_action = input()
        if (train_action == "\n" or train_action == ""):
            train_action = input()

        trainer = GenerateTraner(train_action)
        print_model_menu()
        model_action = input()
        if (model_action == "\n" or model_action == ""):
            model_action = input()

        model_generater = GenerateModel(model_action)
        training_data = TrainingData(config)
        trainer.run_function(model_generater, config, training_data)

        return

    if general_action == 'p':
        print_prediction_menu()
        predic_action = input()
        if (predic_action == "\n" or predic_action == ""):
            predic_action = input()
        predicter = GeneratePredicter(predic_action)
        predicter.run_function(config)
        return

main()