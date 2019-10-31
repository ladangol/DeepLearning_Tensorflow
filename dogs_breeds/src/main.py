import config
from data_prepration import prepare_data
from util import read_labels, get_path

import tensorflow as tf
import numpy as np
from numpy import load
import keras
from sklearn.model_selection import train_test_split

data_placeholder   = tf.compat.v1.placeholder('float', [None, None])
labels_placeholder = tf.compat.v1.placeholder('float32', [None, None])

def define_model(in_config,is_training=False):

    data_reshaped = tf.reshape(data_placeholder,
                               shape=[-1, in_config.image_size, in_config.image_size, in_config.num_channels])

    conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(data_reshaped)
    conv1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, 3, activation=tf.nn.relu)(conv1)
    conv2 = keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = keras.layers.Conv2D(128, 3, activation=tf.nn.relu)(conv2)
    conv3 = keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = keras.layers.Conv2D(128, 3, activation=tf.nn.relu)(conv3)
    conv4 = keras.layers.MaxPooling2D((2, 2))(conv4)

    fc1 = keras.layers.Flatten()(conv4)
    fc1 = keras.layers.Dense(128, activation=tf.nn.relu)(fc1)

    fc1 = keras.layers.Dropout(rate=0.5)(fc1)

    output = keras.layers.Dense(in_config.num_classes)(fc1)

    return output
def load_data(in_config):
    print("Loading data!")
    data_path = get_path(in_config.data_path_root, in_config.data_name)
    labels_path = get_path(in_config.data_path_root, in_config.labels_name)
    data = load(data_path)
    labels = load(labels_path)
    print("Preprocessing data!")
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.07, random_state=42)

    train_y = keras.utils.to_categorical(train_y, in_config.num_classes)
    test_y = keras.utils.to_categorical(test_y, in_config.num_classes)

    train_x = np.reshape(train_x, newshape=[-1, in_config.image_size * in_config.image_size * in_config.num_channels])
    test_x = np.reshape(test_x, newshape=[-1, in_config.image_size * in_config.image_size * in_config.num_channels])

    return train_x, test_x, train_y, test_y

def train_neural_network(in_config):
    (train_x, test_x, train_y, test_y) = load_data(in_config)

    print("Define a Model!")
    logits = define_model(in_config, True)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

    saver = tf.train.Saver()

    print("Start Training....t.!")
    with tf.Session() as sess:
        from keras import backend as K
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())

        for epoch in range(in_config.num_epochs):
            epoch_loss = 0
            i = 0
            while i < train_x.shape[0]:
                start = i
                end = i + in_config.batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={data_placeholder: batch_x, labels_placeholder: batch_y})
                epoch_loss += c
                i += in_config.batch_size
                acc_train = accuracy.eval(feed_dict={data_placeholder: batch_x, labels_placeholder: batch_y})
                print("Epoch:", epoch + 1," Iteration:", int(i/in_config.batch_size) , " Train accuracy:", acc_train)

            print('Epoch', epoch + 1, 'completed out of', in_config.num_epochs, 'loss:', "{0:.3f}".format(epoch_loss) )
            acc_test = accuracy.eval(feed_dict={data_placeholder: test_x, labels_placeholder: test_y})
            print("Epoch:", epoch + 1, " Test accuracy:", acc_test)

            print('saving model')
            model_name = get_path(in_config.model_path_root,  "dog-breeds")
            model_full_path = "{0}-epoch-{1:02d}-epoch_loss-{2:.3f}".format(model_name, epoch+1, epoch_loss)
            saver.save(sess, model_full_path)

def print_main_menu():
    print('press d for data_preparation: ')
    print('press t for train: ')
    print('press p for prediction: ')
    print('press e for exit: ')

def main():
    print_main_menu()
    action = input()
    if action == 'd':
        # define location of dataset
        prepare_data(config)
    elif action == 'e':
        return
    elif action == 't':
        read_labels(config)
        if config.num_classes == -1 or len(config.dict_categories)==0:
            print("Fail to read the class ids, can not proceed!")
            exit(-1)
        print("number of classes: " + str(config.num_classes))
        train_neural_network(config)

    elif action == 'p':
        print("Prediction .....")
        # test_model_path = get_path(config.model_path_root, 'no_cam\\Model-60-0.820.model')
        # test_data_path = get_path(config.data_path_root, 'test')
        # predict(test_data_path, test_model_path, config)


main()