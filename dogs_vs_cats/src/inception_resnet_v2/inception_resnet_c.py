import inception_resnet_v2.conv2d as conv2d

from keras.layers import Lambda
from keras.layers import Concatenate
from keras import backend

def scaling(inputs,scale):
    return inputs[0] + inputs[1] * scale

def increase_c(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d.build(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d.build(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d.build(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d.build(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d.build(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(scaling,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay

def build(x, scale):
    # Inception-ResNet-C modules
    x = increase_c(x, scale, name='increase_c_1')
    x = increase_c(x, scale, name='increase_c_2')
    x = increase_c(x, scale, name='increase_c_3')
    x = increase_c(x, scale, name='increase_c_4')
    x = increase_c(x, scale, name='increase_c_5')

    return x