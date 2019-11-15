import config
from train import train, predict, word2vector_analogy
from data_prepration import prepare_data

def print_main_menu():
    print('press d for data_preparation: ')
    print('press b for build word2vec model: ')
    print('press t for train: ')
    print('press p for prediction: ')
    print('press a for word2vector analogy: ')
    print('press e for exit: ')

def main():
    print_main_menu()
    action = input()
    if action == 'd':
        # define location of dataset
        prepare_data(config)
    elif action == 'b':
        # define location of dataset
        from build_word2vec_model import build_model
        build_model(config)
    elif action == 'e':
        return
    elif action == 't':
        print("Training .....")
        train(config)
    elif action == 'p':
        print("Prediction .....")
        predict(config)
    elif action == 'a':
        print("Analogy .....")
        word2vector_analogy(config)
main()