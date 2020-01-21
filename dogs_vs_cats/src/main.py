import config
from util import get_path, plot_confusion_matrix, get_categories, generate_current_config_to_string
from data_prepration import prepare_data

from simple_cnn import define_model, predict, step_decay
from cam import define_model as define_cam_model, predict as cam_predict


from sklearn.metrics import classification_report

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
from numpy import load, save
import time
import sklearn.metrics as metrics
from keras.callbacks import LearningRateScheduler, CSVLogger
import pandas as pd
import json
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

def train( in_model, in_config, training_data):
    if not isinstance(training_data, TrainingData):
        raise AssertionError("training_data should be provided as a TrainingData!")

    trainX = training_data.trainX
    testX  = training_data.testX
    trainY = training_data.trainY
    testY = training_data.testY

    config_to_string = generate_current_config_to_string(config)
    NAME = 'Cat-vs-dog' + config_to_string + f'{int(time.time())}'
    file_path = "Model-LR"+ str(in_config.initial_lrate) + "-E{epoch:02d}-VA-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    check_point = ModelCheckpoint("Models/logs/{}.model".format(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = get_path(config.model_path_root, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = get_path(log_path, log_file_name)
    tensor_board = TensorBoard(log_dir=log_full_path)
    log_file_full_path = "Models/logs/" + NAME + ".csv"
    csv_logger = CSVLogger(log_file_full_path, append=True, separator=';')

    lrate_scheduler = LearningRateScheduler(step_decay)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    # callback_list = [tensor_board, lrate_scheduler, check_point, csv_logger]
    callback_list = [tensor_board, lrate_scheduler, csv_logger]

    # train the neural network
    history = in_model.fit(trainX, trainY, validation_data=(testX, testY), epochs=in_config.num_epochs,
                 batch_size=in_config.batch_size, verbose=1, callbacks=callback_list)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = in_model.predict(testX, batch_size=in_config.batch_size)
    #create confusion matrix
    # print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=get_categories(in_config)))
    # print("")

    cr = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=get_categories(in_config), output_dict = False)
    confusion_matrix_file_name = get_path(get_path('Models','logs'), in_config.confusion_matrix_detailed_file_name)
    with open(confusion_matrix_file_name, 'w') as f:
        f.write(cr)
    print(cr)
    print("")

    cr = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=get_categories(in_config), output_dict = True)
    confusion_matrix_file_name = get_path(get_path('Models','logs'), 'dic_'+in_config.confusion_matrix_detailed_file_name)
    with open(confusion_matrix_file_name, 'w') as f:
        # print(cr, file=f)
        f.write(json.dumps(cr))

    print(cr)
    print("")

    in_config.grid_serach_validation_result[config_to_string] = cr
    with open("Models/logs/all_test.txt", 'w') as f:
        # print(cr, file=f)
        f.write(json.dumps(in_config.grid_serach_validation_result))

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

        config_to_string = generate_current_config_to_string(config)
        training_data =  TrainingData(config)

        # initial_lrate = [0.1,0.01,0.001,0.0001]
        initial_lrate_list = [0.01,0.001]
        activation_list = ['swish','LeakyReLU','ReLU','Tanh']
        kernel_initializer_list = ['he_uniform','glorot_uniform','lecun_uniform']
        bias_initializer_list = [0.0, 0.01]
        epoch = 30
        for lr in initial_lrate_list:
            for activation in activation_list:
                for kernel in kernel_initializer_list:
                    for bias in bias_initializer_list:
                        config.num_epochs = epoch
                        config.initial_lrate = lr
                        config.kernel_initializer = kernel
                        config.activation = activation
                        config.bias_initializer = bias
                        config.display_plot = False

                        config_to_string = generate_current_config_to_string(config)
                        config.confusion_matrix_detailed_file_name = "confusion_matrix_detailed" + config_to_string +".txt"
                        config.confusion_matrix_file_name = "confusion_matrix" + config_to_string +".txt"
                        config.confusion_matrix_plot_name = "confusion_matrix" + config_to_string +".png"
                        model = None

                        if action == 't':
                            model = define_model(config)
                        elif action == 'e':
                            return
                        elif action == 'c':
                            model = define_cam_model(config.num_classes, config.image_size)
                        if model != None:
                            train(model, config, training_data)

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