import inception_resnet_v2.conv2d as conv2d

from keras.layers import Concatenate
from keras.layers import MaxPooling2D

def build(input):
    x = conv2d.build(input, 32, 3, 2, 'valid', True, name='conv1')
    x = conv2d.build(x, 32, 3, 1, 'valid', True, name='conv2')
    x = conv2d.build(x, 64, 3, 1, 'valid', True, name='conv3')

    x_11 = MaxPooling2D(3, strides=1, padding='valid', name='stem_br_11' + '_maxpool_1')(x)
    x_12 = conv2d.build(x, 64, 3, 1, 'valid', True, name='stem_br_12')

    x = Concatenate(axis=3, name='stem_concat_1')([x_11, x_12])

    x_21 = conv2d.build(x, 64, 1, 1, 'same', True, name='stem_br_211')
    x_21 = conv2d.build(x_21, 64, [1, 7], 1, 'same', True, name='stem_br_212')
    x_21 = conv2d.build(x_21, 64, [7, 1], 1, 'same', True, name='stem_br_213')
    x_21 = conv2d.build(x_21, 96, 3, 1, 'valid', True, name='stem_br_214')

    x_22 = conv2d.build(x, 64, 1, 1, 'same', True, name='stem_br_221')
    x_22 = conv2d.build(x_22, 96, 3, 1, 'valid', True, name='stem_br_222')

    x = Concatenate(axis=3, name='stem_concat_2')([x_21, x_22])

    x_31 = conv2d.build(x, 192, 3, 1, 'valid', True, name='stem_br_31')
    x_32 = MaxPooling2D(3, strides=1, padding='valid', name='stem_br_32' + '_maxpool_2')(x)
    x = Concatenate(axis=3, name='stem_concat_3')([x_31, x_32])

    return x