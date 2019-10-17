import config
from util import get_path, plot_confusion_matrix, get_categories
from data_prepration import prepare_data

from simple_cnn import define_model, predict
from cam import define_model as define_cam_model, predict as cam_predict

import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
from numpy import load
import time
import sklearn.metrics as metrics

def train( in_model, in_config):
    print("Loading data!")
    data_path = get_path(in_config.data_path_root, in_config.data_name)
    labels_path = get_path(in_config.data_path_root, in_config.labels_name)
    data = load(data_path)
    labels = load(labels_path)

    print("Preprocessing data!")
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, in_config.num_classes)
    testY = keras.utils.to_categorical(testY,  in_config.num_classes)

    NAME = f'Cat-vs-dog-cnn-64x2-{int(time.time())}'
    file_path = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    check_point = ModelCheckpoint("Models/{}.model".format(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = get_path(config.model_path_root, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = get_path(log_path, log_file_name)
    tensor_board = TensorBoard(log_dir=log_full_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    callback_list = [check_point, tensor_board]

    # train the neural network
    in_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=in_config.num_epochs,
                 batch_size=in_config.batch_size, verbose=1, callbacks=callback_list)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = in_model.predict(testX, batch_size=in_config.batch_size)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=get_categories(in_config)))
    print("")

    y_pred_classes = np.argmax(predictions, axis=1)
    y_true = np.argmax(testY, axis=1)
    confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_classes)

    plot_confusion_matrix(confusion_matrix, in_config)

def print_main_menu():
    print('press d for data_preparation: ')
    print('press t for train: ')
    print('press p for prediction: ')
    print('press e for exit: ')

def print_train_menu():
    print('press t for train: ')
    print('press c for cam train: ')
    print('press e for exit: ')

def print_prediction_menu():
    print('press p for prediction: ')
    print('press c for cam prediction: ')
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
        print_train_menu()
        action = input()
        if (action == "\n" or action == ""):
            action = input()
        model = None
        if action == 't':
            model = define_model(config.num_classes, config.image_size)
        elif action == 'e':
            return
        elif action == 'c':
            model = define_cam_model(config.num_classes, config.image_size)
        if model != None:
            train(model, config)

    elif action == 'p':
        print_prediction_menu()
        action = input()
        if (action == "\n" or action == ""):
            action = input()
        if action == 'p':
            test_model_path = get_path(config.model_path_root, 'no_cam\\Model-60-0.820.model')
            test_data_path = get_path(config.data_path_root, 'test')
            predict(test_data_path, test_model_path, config)
        elif action == 'e':
            return
        elif action == 'c':
            test_model_path = get_path(config.model_path_root, 'Vgg_16_Cam\\Model-02-0.978.model')
            test_data_path = get_path(config.data_path_root, 'test\\cam')
            cam_predict(test_data_path, test_model_path, config.image_size)

main()