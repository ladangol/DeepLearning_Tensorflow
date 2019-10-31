from util import get_path, get_category
from numpy import load
from model import AnimalTypeBreeds
# import the necessary packages
from keras.optimizers import Adam
import keras
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


def train(in_config):
    print("Loading data!")
    data_path = get_path(in_config.data_path_root, in_config.data_name)
    labels_path = get_path(in_config.data_path_root, in_config.labels_name)
    data = load(data_path)
    type_labels = load(labels_path)
    breeds_labels = load(labels_path)

    print("Preprocessing data!")
    split = train_test_split(data, type_labels, breeds_labels,
        test_size=0.2, random_state=42)
    (trainX, testX, train_typesY, test_typesY,
        train_breedsY, test_breedsY) = split

    train_typesY = keras.utils.to_categorical(train_typesY, in_config.num_types)
    test_typesY = keras.utils.to_categorical(test_typesY,  in_config.num_types)
    train_breedsY = keras.utils.to_categorical(train_breedsY, in_config.num_breeds)
    test_breedsY = keras.utils.to_categorical(test_breedsY,  in_config.num_breeds)

    model = AnimalTypeBreeds.build(in_config, in_finalAct="softmax")

    losses = {
        "type_output": "categorical_crossentropy",
        "breed_output": "categorical_crossentropy",
    }
    lossWeights = {"type_output": 1.0, "breed_output": 1.0}

    print("[INFO] compiling model...")
    opt = Adam(lr=in_config.INIT_LR, decay=in_config.INIT_LR / in_config.EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
        metrics=["accuracy"])

    H = model.fit(trainX,
        {"type_output": train_typesY, "breed_output": train_breedsY},
        validation_data=(testX,
            {"type_output": test_typesY, "breed_output": test_breedsY}),
        epochs=in_config.EPOCHS,
        verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model_name = get_path(in_config.model_path_root, "dog_cat_breed.model")
    model.save(model_name)


def predict(in_data_path, in_model_path, in_config ):
    image_size = in_config.image_size
    model = keras.models.load_model(in_model_path)
    for image_name in os.listdir(in_data_path):
        image_path = get_path(in_data_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))

        print("[INFO] classifying image...")
        (types_Proba, breeds_Proba) = model.predict([image.reshape(-1, image_size, image_size, 3)])

        image = cv2.imread(image_path)
        types_Idx = np.argmax(types_Proba)
        types_label = get_category(in_config, types_Idx)

        breeds_idx = np.argmax(breeds_Proba)
        breeds_label = get_category(in_config, breeds_idx)

        types_text = "Type: {} ({:.2f}%)".format(types_label,
                                                       types_Proba[0][types_Idx] * 100)
        breeds_text = "color: {} ({:.2f}%)".format(breeds_label,
                                                 breeds_Proba[0][breeds_idx] * 100)
        cv2.putText(image, types_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(image, breeds_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        print("[INFO] {}".format(types_text))
        print("[INFO] {}".format(breeds_text))

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)