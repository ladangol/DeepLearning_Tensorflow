
    #Augmentation
    # if you have less than 25K you shoudl do it, do it after data spliting. Augmentation is performed only on train data.
      # Augment(do_augment=False, save=Fale)

    # improvement:

    # in which order we should tune hyperparameters in neural network ?????

    # OP1:
    # add scheduler (we must have learning rate decay)
        # Start LR = = 0.01
        # 30 epoch,
        # start: epoch #10: LR = add 0.1 to LR and every 5 epoch you add 0.1 to LR
        # Note:
        # in our case if we reatch to minimum at epoch 6 then starting_drop  should set to epoch 4 or epoch 5

    # OP2:
    # if you do not using TL then you need to optimize kernel_initializer and bias_initializer (???)
        #glorot_normal(seed = NULL)  ->Tanh
        #glorot_uniform(seed = NULL) ->Tanh
        #he_normal(seed = NULL) -> Relu
        #he_uniform(seed = NULL) -> Relu
        #lecun_normal(seed=None) -> SELU
        #lecun_uniform(seed=None) -> SELU

    # OP3:
    # if you do not using TL then you need to optimize activation function
        # https://en.wikipedia.org/wiki/Activation_function
        # https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
        # TanH
        # Rectified linear unit (ReLU)
        # Leaky rectified linear unit (Leaky ReLU)
        # Inverse square root linear unit (ISRLU)
        # SWISH: Sigmoid Linear Unit (SiLU)(AKA SiL and Swish-1)

    # OP4:
    # model architecture modification: add layer then you have to do Op1 till Op3 for new model

    # model evaluation
    #metrics ????



from util import get_path, get_category

import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers import LeakyReLU, PReLU, ReLU

from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import numpy as np
import os
import cv2
import math

import matplotlib.pyplot as plt

# learning rate schedule
def step_decay(epoch):
    from  config import initial_lrate
    print("in epoch#: " ,epoch, " initial lrate: ",initial_lrate)
    drop = 0.1
    epochs_drop = 10
    start_drop = 10
    if start_drop < epoch:
        epochs_drop = 5

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate

def add_swish_activation():
    from keras import backend as K

    def swish_activation(x):
            return (K.sigmoid(x) * x)

    from keras.utils.generic_utils import get_custom_objects
    from keras.layers import Activation
    get_custom_objects().update({'swish': Activation(swish_activation)})

def get_activation(in_config):
    switcher = {
            'ReLU': ReLU(max_value=None, negative_slope=0.0, threshold=0.0),
            'Tanh': "tanh",
            'LeakyReLU': LeakyReLU(alpha=0.3),
            'PReLU': PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None),
            'swish': "swish"
        }
    return switcher.get(in_config.activation, "invalid")

def define_model(in_config):
    add_swish_activation()
    model = Sequential()
    activation  = get_activation(in_config)
    kernel_init = in_config.kernel_initializer
    bias        = keras.initializers.Constant(value=in_config.bias_initializer)

    model.add(Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, bias_initializer=bias, padding='same', input_shape=(in_config.image_size, in_config.image_size, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, bias_initializer=bias, padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, bias_initializer=bias, padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, bias_initializer=bias, padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation=activation, kernel_initializer=kernel_init, bias_initializer=bias))

    model.add(Dropout(0.5))

    model.add(Dense(in_config.num_classes, activation="softmax"))
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
        if in_config.display_plot:
            plt.show()
