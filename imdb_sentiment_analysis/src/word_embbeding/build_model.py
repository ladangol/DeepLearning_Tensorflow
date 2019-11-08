from src.util import  get_path
from keras.models import Sequential
from keras.layers import Dense, GRU, Flatten, Embedding

def define_model(in_config):
    info_file_name = get_path(in_config.data_path_root, in_config.info_file_name)
    file = open(info_file_name, "r")
    vocab_size = int(file.readline())
    max_length = int(file.readline())
    file.close()

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=in_config.output_dim, input_length=max_length))
    if in_config.add_gru_layer:
        model.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    else:
        model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())

    return model