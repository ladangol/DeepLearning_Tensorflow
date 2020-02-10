import inception_resnet_v2.inception_resnet_a as block_a
import inception_resnet_v2.inception_resnet_b as block_b
import inception_resnet_v2.inception_resnet_c as block_c
import inception_resnet_v2.stem as stem

from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD

def define_model(config):
    input = Input(shape=(config.image_size, config.image_size, config.num_channel))

    x = stem.build(input)
    # Inception-ResNet-A modules
    x = block_a.build(x, 0.15)
    # Inception-ResNet-B modules
    x = block_b.build(x, 0.1)
    # Inception-ResNet-C modules
    x = block_c.build(x, 0.2)

    # TOP
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    output = Dense(config.num_classes, activation='softmax')(x)

    model = Model(input, output, name='inception_resnet_v2')

    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model .summary()
    return model
