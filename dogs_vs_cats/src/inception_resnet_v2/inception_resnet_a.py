import inception_resnet_v2.conv2d as conv2d

from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras import backend

def scaling(inputs,scale):
    return inputs[0] + inputs[1] * scale

def increase_a(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d.build(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d.build(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d.build(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d.build(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d.build(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d.build(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d.build(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(scaling,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def reduction_a(x):
    # 35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_1')(x)

    x_red_12 = conv2d.build(x, 384, 3, 2, 'valid', True, name='x_red1_c1')

    x_red_13 = conv2d.build(x, 256, 1, 1, 'same', True, name='x_red1_c2_1')
    x_red_13 = conv2d.build(x_red_13, 256, 3, 1, 'same', True, name='x_red1_c2_2')
    x_red_13 = conv2d.build(x_red_13, 384, 3, 2, 'valid', True, name='x_red1_c2_3')
    x = Concatenate(axis=3, name='red_concat_1')([x_red_11, x_red_12, x_red_13])
    return x

def build(x, scale):
    # Inception-ResNet-A modules
    x = increase_a(x, scale, name='increase_a_1')
    x = increase_a(x, scale, name='increase_a_2')
    x = increase_a(x, scale, name='increase_a_3')
    x = increase_a(x, scale, name='increase_a_4')
    x = increase_a(x, scale, name='increase_a_5')

    return reduction_a(x)