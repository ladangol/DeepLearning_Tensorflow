import inception_resnet_v2.conv2d as conv2d

from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras import backend

def scaling(inputs,scale):
    return inputs[0] + inputs[1] * scale

def increase_b(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d.build(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d.build(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d.build(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d.build(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d.build(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(scaling,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def reduction_b(x):
    # 17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_2')(x)

    x_red_22 = conv2d.build(x, 256, 1, 1, 'same', True, name='x_red2_c11')
    x_red_22 = conv2d.build(x_red_22, 384, 3, 2, 'valid', True, name='x_red2_c12')

    x_red_23 = conv2d.build(x, 256, 1, 1, 'same', True, name='x_red2_c21')
    x_red_23 = conv2d.build(x_red_23, 256, 3, 2, 'valid', True, name='x_red2_c22')

    x_red_24 = conv2d.build(x, 256, 1, 1, 'same', True, name='x_red2_c31')
    x_red_24 = conv2d.build(x_red_24, 256, 3, 1, 'same', True, name='x_red2_c32')
    x_red_24 = conv2d.build(x_red_24, 256, 3, 2, 'valid', True, name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21, x_red_22, x_red_23, x_red_24])
    return x


def build(x, scale):
    # Inception-ResNet-B modules
    x = increase_b(x, scale, name='increase_b_1')
    x = increase_b(x, scale, name='increase_b_2')
    x = increase_b(x, scale, name='increase_b_3')
    x = increase_b(x, scale, name='increase_b_4')
    x = increase_b(x, scale, name='increase_b_5')
    x = increase_b(x, scale, name='increase_b_6')
    x = increase_b(x, scale, name='increase_b_7')
    x = increase_b(x, scale, name='increase_b_8')
    x = increase_b(x, scale, name='increase_b_9')
    x = increase_b(x, scale, name='increase_b_10')

    return reduction_b(x)