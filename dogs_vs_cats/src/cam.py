from util import getPath

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D

import scipy as sp
from matplotlib import pyplot
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def define_model_cam(in_num_classes, in_image_size):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(in_image_size, in_image_size, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='lastCov'))
    model.add(AveragePooling2D((9,9), name='GAP'))

    model.add(Flatten())

    #weight
    model.add(Dense(in_num_classes, activation="softmax"))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model .summary()
    return model

from keras.applications.vgg16 import VGG16

# define cnn model
def define_VGG16_model_cam(in_num_classes, in_image_size):
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = True
	# add new classifier layers
	GAP = AveragePooling2D(14,14)(model.layers[-2].output)
	flat1 = Flatten()(GAP)
	output = Dense(in_num_classes, activation='softmax')(flat1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)

    # binary_accuracy or categorical_accuracy
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	model .summary()
	return model

def predic_cam(in_data_path, in_model_path, in_image_size):
    model = keras.models.load_model(in_model_path)
    # get the weights from the last layer
    gap_weights = model.layers[-1].get_weights()[0]
    model.summary()
    # create a new model to output the feature maps and the predicted labels
    cam_model = Model(inputs=model.input,
                      outputs=(model.get_layer("block5_conv3").output, model.layers[-1].output))
    cam_model.summary()

    for image_name in os.listdir(in_data_path):
        idx = 0
        image_path = getPath(in_data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (in_image_size, in_image_size))

        features, results = cam_model.predict([image.reshape(-1, in_image_size, in_image_size, 3)])

        # get the feature map of the test image
        features_for_one_img = features[idx, :, :, :]

        # map the feature map to the original size
        height_roomout = in_image_size / features_for_one_img.shape[0]
        width_roomout = in_image_size / features_for_one_img.shape[1]
        # over sample the heat map to overlay on image
        cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=2)

        # get the predicted label with the maximum probability
        pred = np.argmax(results[idx])

        # prepare the final display
        plt.figure(facecolor='white')

        # get the weights of class activation map
        cam_weights = gap_weights[:, pred]

        # create the class activation map
        cam_output = np.dot(cam_features, cam_weights)

        # draw the class activation map
        # plot first few images
        categories = ["Dog", "Cat"]
        label = 'Image name: ' + image_name +', Predicted Class = ' + categories[pred] + ', Probability = ' + str(results[idx][pred])
        pyplot.subplot(121)
        pyplot.imshow(image)
        pyplot.title('Original Image')
        pyplot.subplot(122)
        pyplot.imshow(image)
        pyplot.imshow(cam_output, cmap='jet', alpha=0.5)
        pyplot.suptitle(label, fontsize=10)
        pyplot.title('Class Activation Map')
        pyplot.show()

