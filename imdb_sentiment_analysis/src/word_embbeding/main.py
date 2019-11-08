import config
from train import train, predict
from data_prepration import prepare_data

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
        prepare_data(config)
    elif action == 'e':
        return
    elif action == 't':
        print("Training .....")
        train(config)
    elif action == 'p':
        print("Prediction .....")
        predict(config)
main()