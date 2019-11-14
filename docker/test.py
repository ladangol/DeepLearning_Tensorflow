
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

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

def main():
	print("test my code")
	import tensorflow as tf
	hello = tf.constant("hello TensorFlow!")
	sess=tf.Session()

	from tensorflow.python.client import device_lib
	local_device_protos = device_lib.list_local_devices()
	print(sess.run(hello))
	for x in local_device_protos:
		print(x.name)
	print("press enter to continue!")
	action = input()
	print("Defining a model")
	define_model(2,224)
	print("test is done")

main()