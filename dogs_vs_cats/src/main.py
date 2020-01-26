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

class GenerateModel:
    def __init__(self,model_type):
        self.fActive = self.generate_model(model_type) # mapping: string --> variable = function name

    def generate_model(self, model_type):
        switcher = {
            's': define_model,
            'c': define_cam_model
        }

        current_model = switcher.get(model_type.lower(), None)

        if current_model == None:
            raise AssertionError("model_type is invalid!")

        return current_model

    def run_function(self, config):
        return  self.fActive(config)

class GenerateTraner:
    def __init__(self,training_type):
        self.fActive = self.generate_trainer(training_type) # mapping: string --> variable = function name

    def generate_trainer(self, training_type):
        switcher = {
            'gs': grind_serach,
            't': train
        }

        trainer = switcher.get(training_type.lower(), None)

        if trainer == None:
            raise AssertionError("training_type is invalid!")

        return trainer

    def run_function(self, in_model_generater, in_config, training_data):
        self.fActive(in_model_generater, in_config, training_data)

class GeneratePredicter:
    def __init__(self, predicting_type):
        self.fActive = self.generate_predicter(predicting_type) # mapping: string --> variable = function name

    def generate_predicter(self, predicting_type):
        switcher = {
            'p': self.predict,
            'c': self.cam_predict
        }

        predicter = switcher.get(predicting_type.lower(), None)

        if predicter == None:
            raise AssertionError("predicting_type is invalid!")

        return predicter

    def predict(self):
            test_model_path = get_path(config.model_path_root, 'no_cam\\Model-60-0.820.model')
            test_data_path = get_path(config.data_path_root, 'test')
            predict(test_data_path, test_model_path, config)
    def cam_predict(self, config):
        test_model_path = get_path(config.model_path_root, 'Vgg_16_Cam\\Model-02-0.978.model')
        test_data_path = get_path(config.data_path_root, 'test\\cam')
        cam_predict(test_data_path, test_model_path, config.image_size)

    def run_function(self, config):
        self.fActive(config)

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

def grind_serach(in_model_generater,in_config, training_data):
    if in_model_generater == None:
        return
    config_to_string = generate_current_config_to_string(config)

    # initial_lrate = [0.1,0.01,0.001,0.0001]
    initial_lrate_list = [0.01, 0.001]
    activation_list = ['ReLU', 'swish', 'LeakyReLU', 'Tanh']
    kernel_initializer_list = ['he_uniform', 'glorot_uniform', 'lecun_uniform']
    bias_initializer_list = [0.0, 0.01]
    epoch = 1
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
                    config.confusion_matrix_detailed_file_name = "confusion_matrix_detailed" + config_to_string + ".txt"
                    config.confusion_matrix_file_name = "confusion_matrix" + config_to_string + ".txt"
                    config.confusion_matrix_plot_name = "confusion_matrix" + config_to_string + ".png"

                    train(in_model_generater, config, training_data)

    grid_serch_result_path = get_path(config.data_path_root, 'all_test.txt')
    grid_search = {}
    with open(grid_serch_result_path, "r") as content:
        grid_search = json.load(content)

    new_dict = {'name': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    for key, val in grid_search.items():
        new_dict.get('name').append(key)
        new_dict.get('accuracy').append(val.get('accuracy'))
        weighted_avg_dict = val.get('weighted avg')
        new_dict.get('precision').append(weighted_avg_dict.get('precision'))
        new_dict.get('recall').append(weighted_avg_dict.get('recall'))
        new_dict.get('f1-score').append(weighted_avg_dict.get('f1-score'))
    df = pd.DataFrame.from_dict(new_dict)
    sorted_df = df.sort_values('f1-score')
    # find max frequency of the gs with accuracy about 85
    # start filtering with thoes have maximum frequency
    worst_results = sorted_df.iloc[0:5, :]
    best_results = sorted_df.iloc[-5:, :]
    result_full_path = get_path(config.data_path_root, 'worth_results.csv')
    worst_results.to_csv(result_full_path)
    result_full_path = get_path(config.data_path_root, 'best_results.csv')
    best_results.to_csv(result_full_path)


def train( in_model_generater, in_config, training_data):
    if not isinstance(training_data, TrainingData):
        raise AssertionError("training_data should be provided as a TrainingData!")

    trainX = training_data.trainX
    testX  = training_data.testX
    trainY = training_data.trainY
    testY = training_data.testY

    in_model = in_model_generater.run_function(config)

    config_to_string = generate_current_config_to_string(config)
    NAME = 'Cat-vs-dog' + config_to_string + f'{int(time.time())}'
    file_path = "Model-LR"+ str(in_config.initial_lrate) + "-E{epoch:02d}-VA-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    check_point = ModelCheckpoint("Models/logs/{}.model".format(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = get_path(config.model_path_root, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = get_path(log_path, log_file_name)
    # tensor_board = TensorBoard(log_dir=log_full_path)
    # save class labels to disk to color data points in TensorBoard accordingly
    from os import makedirs
    from os.path import exists, join
    if not exists("Models/logs"):
        makedirs("Models/logs")

    with open('Models/logs/metadata.tsv', 'w') as f:
        np.savetxt(f, np.zeros(len(testY)))

    tensor_board = TensorBoard(log_dir=log_full_path,
                              batch_size=in_config.batch_size,
                              embeddings_freq=1,
                              embeddings_layer_names=['features'],
                              embeddings_metadata='metadata.tsv',
                              embeddings_data=testX)

    log_file_full_path = "Models/logs/" + NAME + ".csv"
    csv_logger = CSVLogger(log_file_full_path, append=True, separator=';')

    lrate_scheduler = LearningRateScheduler(step_decay)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

    callback_list = []
    if config.display_plot == True:
         callback_list = [tensor_board, lrate_scheduler, check_point,csv_logger]
    else:
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