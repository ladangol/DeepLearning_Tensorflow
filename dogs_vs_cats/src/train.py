import config
from util import get_path, plot_confusion_matrix, get_categories, generate_current_config_to_string
from main import TrainingData
from simple_cnn import step_decay

from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
import numpy as np
import pandas as pd
import json
import time



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

