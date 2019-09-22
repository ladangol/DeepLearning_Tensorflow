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

data_dir = os.path.join("data","finalize_dogs_vs_cats")
image_size = 224
EPOCHS = 150
num_classes = 2

save_model = 'Models\simple_nn.model'
save_label = 'Models\simple_nn_lb.pickle'
save_plot = 'Models\simple_nn_plot.png'
data_path = 'data/dogs-vs-cats'
photoes_name = os.path.join(data_path, 'simple_dogs_vs_cats_photos.npy')
labels_name = os.path.join(data_path, 'simple_dogs_vs_cats_labels.npy')

def prepare_data(in_data_dir, in_image_size):
    imagePaths = []

    # define location of dataset
    folder = os.path.join(data_path, 'train/')

    # enumerate files in the directory
    for file in listdir(folder):
        imagePath = os.path.join(folder, file)  # create path to dogs and cats
        imagePaths.append(imagePath)

    random.seed(42)
    random.shuffle(imagePaths)
    data, labels = list(), list()
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (in_image_size, in_image_size))
        data.append(image)
        #label = imagePath.split(os.path.sep)[-2]

        # determine class
        output = 0.0
        label = os.path.basename(imagePath)
        if label.lower().startswith('cat'):
            output = 1.0

        labels.append(output)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    print(data.shape, labels.shape)
    # save the reshaped photos
    save(photoes_name, data)
    save(labels_name, labels)

    return data, labels


def define_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))

   #model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    #model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    #model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)
    #opt='adam'
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model .summary()
    return model


def predict_multi():
    categories = ["Dog", "Cat"]
    model_path = 'Models\\Model-48-0.814.model'
    model = keras.models.load_model(model_path)
    data_path = 'data\\dogs-vs-cats\\test'
    for image_name in os.listdir(data_path):
        image_path = os.path.join(data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))

        predictions = model.predict([image.reshape(-1, image_size, image_size, 3)])
        classId = np.argmax(predictions)
        className = categories[classId]
        print(image_name + ': Prediction ' + className)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(className)
        plt.show()


def main():
    do_predict = True
    if (do_predict):
        predict_multi()
        return

    do_data_preparation = False
    if(do_data_preparation):
        data, labels = prepare_data(data_dir, image_size)

    data = load(photoes_name)
    labels = load(labels_name)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY,  num_classes)

    model = define_model()

    NAME = f'Cat-vs-dog-cnn-64x2-{int(time.time())}'
    filepath = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("Models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    tensorBoard = TensorBoard(log_dir='Models\logs\{}'.format(NAME))

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

    # # plot the training loss and accuracy
    # N = np.arange(0, EPOCHS)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(N, H.history["loss"], label="train_loss")
    # plt.plot(N, H.history["val_loss"], label="val_loss")
    # plt.plot(N, H.history["acc"], label="train_acc")
    # plt.plot(N, H.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy (Simple NN)")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.savefig(save_plot)
    #
    # # save the model and label binarizer to disk
    # print("[INFO] serializing network and label binarizer...")

    model.save(save_model)

main()