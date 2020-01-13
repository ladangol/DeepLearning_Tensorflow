    # improvement:

    # OP1:
    # add scheduler (we must have learning rate decay)
    # Start LR = = 0.000001
    # 30 epoch,
    # start: epoch #10: LR = 0.00001 and every 5 epoch you remove 0.1 from LR

    # OP2:
    # if you do not using TL then you need to optimize kernel_initializer

    # OP3:
    # if you do not using TL then you need to optimize activation function

    # OP4:
    # model architecture modification: add layer then you have to do Op1 till Op3 for new model
    
from util import get_path, get_category

import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import numpy as np
import os
import cv2

import matplotlib.pyplot as plt


def define_model(in_num_classes, in_image_size):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(in_image_size, in_image_size, 3)))
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

    model.add(Dense(in_num_classes, activation="softmax"))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model .summary()
    return model

def predict(in_data_path, in_model_path, in_config ):
    image_size = in_config.image_size
    model = keras.models.load_model(in_model_path)
    for image_name in os.listdir(in_data_path):
        image_path = get_path(in_data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))

        predictions = model.predict([image.reshape(-1, image_size, image_size, 3)])
        class_id = np.argmax(predictions)
        class_name = get_category(in_config, class_id)
        print(image_name + ': Prediction ' + class_name)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_name)
        plt.show()
