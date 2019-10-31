# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input

class AnimalTypeBreeds:
    @staticmethod
    def build_animal_types_branch(in_inputs, in_num_types, in_finalAct="softmax", in_chanDim=-1):
        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(in_inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(in_num_types)(x)
        x = Activation(in_finalAct, name="type_output")(x)
        # return the category prediction sub-network
        return x

    @staticmethod
    def build_animal_breeds_branch(in_inputs, in_num_breeds, in_finalAct="softmax", in_chanDim=-1):
        # CONV => RELU => POOL
        x = Conv2D(16, (3, 3), padding="same")(in_inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=in_chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        # define a branch of output layers for the number of different
        # colors (i.e., red, black, blue, etc.)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(in_num_breeds)(x)
        x = Activation(in_finalAct, name="breed_output")(x)

        # return the color prediction sub-network
        return x

    @staticmethod
    def build(in_config, in_finalAct="softmax"):
        inputShape = (in_config.image_size, in_config.image_size, in_config.num_channels)
        in_chanDim = -1

        inputs = Input(shape=inputShape)
        types_branch = AnimalTypeBreeds.build_animal_types_branch(inputs,
            in_config.num_types, in_finalAct=in_finalAct, in_chanDim=in_chanDim)
        breeds_branch = AnimalTypeBreeds.build_animal_breeds_branch(inputs,
            in_config.num_breeds, in_finalAct=in_finalAct, in_chanDim=in_chanDim)

        model = Model(
            inputs=inputs,
            outputs=[types_branch, breeds_branch],
            name="AnimalTypeBreeds")

        return model