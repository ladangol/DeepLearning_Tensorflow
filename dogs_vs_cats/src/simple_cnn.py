from util import getPath
from dogs_vs_cats.src.cam import *

import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from os import listdir
from numpy import save, load
import time

image_size = 224
EPOCHS = 150
categories = ["Dog", "Cat"]
num_classes = 2

model_path = 'models'
data_path = 'data'

save_model = getPath(model_path,'simple_nn.model')
save_label = getPath(model_path,'simple_nn_lb.pickle')
save_plot = getPath(model_path,'simple_nn_plot.png')

photoes_name = getPath(data_path, 'simple_dogs_vs_cats_photos.npy')
labels_name = getPath(data_path, 'simple_dogs_vs_cats_labels.npy')


do_data_preparation = False
do_train = False
do_predict = False
do_cam_train = False
do_cam_predict = True

def prepare_data(in_data_dir, in_image_size):
    imagePaths = []

    # enumerate files in the directory
    for file in listdir(in_data_dir):
        imagePath = getPath(in_data_dir, file)  # create path to dogs and cats
        imagePaths.append(imagePath)

    random.seed(42)
    random.shuffle(imagePaths)
    data, labels = list(), list()
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (in_image_size, in_image_size))
        data.append(image)

        # determine class
        output = float(categories.index("Dog"))
        label = os.path.basename(imagePath)
        if label.lower().startswith('cat'):
            output = float(categories.index("Cat"))

        labels.append(output)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # save the reshaped photos
    save(photoes_name, data)
    save(labels_name, labels)

    print("Data preparation completed!")
    return data, labels


def define_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model .summary()
    return model



def predict(in_data_path, in_model_path):
    model = keras.models.load_model(in_model_path)
    for image_name in os.listdir(in_data_path):
        image_path = getPath(in_data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))

        predictions = model.predict([image.reshape(-1, image_size, image_size, 3)])
        classId = np.argmax(predictions)
        className = categories[classId]
        print(image_name + ': Prediction ' + className)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(className)
        plt.show()

def train():
    print("Loading data!")
    data = load(photoes_name)
    labels = load(labels_name)

    print("Preprocessing data!")
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY,  num_classes)

    model = define_model()

    NAME = f'Cat-vs-dog-cnn-64x2-{int(time.time())}'
    filepath = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("Models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = getPath(model_path, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = getPath(log_path, log_file_name)
    tensorBoard = TensorBoard(log_dir=log_full_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    callback_list = [checkpoint, tensorBoard]

    # train the neural network
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  epochs=EPOCHS, batch_size=32, verbose=1, callbacks=callback_list)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=num_classes))

    model.save(save_model)


def train_cam():
    model = define_VGG16_model_cam(num_classes, image_size)
    # model = define_model_cam(num_classes, image_size)

    print("Loading data!")
    data = load(photoes_name)
    labels = load(labels_name)

    print("Preprocessing data!")
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY,  num_classes)



    NAME = f'Cat-vs-dog-cnn-64x2-{int(time.time())}'
    filepath = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("Models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = getPath(model_path, 'logs')
    log_file_name = '{}'.format(NAME)
    log_full_path = getPath(log_path, log_file_name)
    tensorBoard = TensorBoard(log_dir=log_full_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    callback_list = [checkpoint, tensorBoard]

    # train the neural network
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  epochs=EPOCHS, batch_size=32, verbose=1, callbacks=callback_list)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=num_classes))

    model.save(save_model)
def main():
    if(do_data_preparation):
        # define location of dataset
        train_data_path = getPath(data_path, 'train')
        prepare_data(train_data_path, image_size)

    if(do_train):
        train()

    if (do_predict):
        test_model_path = getPath(model_path,'Model-60-0.820.model')
        test_data_path = getPath(data_path, 'test')
        predict(test_data_path, test_model_path)

    if (do_cam_train):
        train_cam()

    if (do_cam_predict):
        test_model_path = getPath(model_path, 'Model-02-0.978.model')
        test_data_path = getPath(data_path, 'test')
        predic_cam(test_data_path, test_model_path, image_size)

main()