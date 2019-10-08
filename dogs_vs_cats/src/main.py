import config
from util import getPath
from data_prepration import prepare_data

from simple_cnn import define_model, predict
from cam import define_model as define_cam_model, predict as cam_predict

import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from numpy import load
import time


def train( in_model, in_config):
    in_num_classes = in_config.num_classes
    in_batch_size = in_config.num_classes
    in_num_epochs = in_config.num_classes
    in_data_path = getPath(in_config.data_path_root, in_config.data_name)
    in_labels_path = getPath(in_config.data_path_root, in_config.labels_name)

    print("Loading data!")
    data = load(in_data_path)
    labels = load(in_labels_path)

    print("Preprocessing data!")
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, in_num_classes)
    testY = keras.utils.to_categorical(testY,  in_num_classes)

    NAME = f'Cat-vs-dog-cnn-64x2-{int(time.time())}'
    file_path = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    check_point = ModelCheckpoint("Models/{}.model".format(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = getPath(config.model_path_root, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = getPath(log_path, log_file_name)
    tensor_board = TensorBoard(log_dir=log_full_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    callback_list = [check_point, tensor_board]

    # train the neural network
    in_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=in_num_epochs,
                 batch_size=in_batch_size, verbose=1, callbacks=callback_list)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = in_model.predict(testX, batch_size=in_batch_size)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=in_num_classes))

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
    print('press t for prediction: ')
    print('press c for cam prediction: ')
    print('press e for exit: ')

def main():
    do_data_preparation = False
    do_train = False
    do_predict = False
    do_cam_train = False
    do_cam_predict = False

    print_main_menu()
    action = input()
    if action == 'd':
        do_data_preparation = True
    elif action == 'e':
        return
    elif action == 't':
        print_train_menu()
        action = input()
        if action == 't':
            do_train = True
        elif action == 'e':
            return
        elif action == 'c':
            do_cam_train = True

    elif action == 'p':
        print_prediction_menu()
        action = input()
        if action == 'p':
            do_predict = True
        elif action == 'e':
            return
        elif action == 'c':
            do_cam_predict = True

    if(do_data_preparation):
        # define location of dataset
        train_data_path = getPath(config.data_path_root, 'train')
        prepare_data(train_data_path, config)

    if(do_train):
        model = define_model(config.num_classes, config.image_size)
        train(model, config)

    if (do_predict):
        test_model_path = getPath(config.model_path_root,'Model-60-0.820.model')
        test_data_path = getPath(config.data_path_root, 'test')
        predict(test_data_path, test_model_path)

    if (do_cam_train):
        model = define_cam_model(config.num_classes, config.image_size)
        train(model, config)

    if (do_cam_predict):
        test_model_path = getPath(config.model_path_root, 'Vgg_16_Cam\\Model-02-0.978.model')
        test_data_path = getPath(config.data_path_root, 'test\\cam')
        cam_predict(test_data_path, test_model_path, config.image_size)

main()