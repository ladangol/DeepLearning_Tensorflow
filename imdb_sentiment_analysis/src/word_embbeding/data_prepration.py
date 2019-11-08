from src.util import  get_path

import csv
from random import shuffle
from numpy import save, array
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def data_encoding(X_train, X_test):
    tokenizer_obj = Tokenizer()
    total_reviewer = X_train + X_test
    tokenizer_obj.fit_on_texts(total_reviewer)

    max_length = max([len(s.split()) for s in total_reviewer])
    vocab_size = len(tokenizer_obj.word_index) + 1

    X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
    X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

    return X_train_pad, X_test_pad, vocab_size, max_length

def prepare_data(in_config):
    print("Start preparing data!")
    file_name = get_path(in_config.data_path_root, in_config.data_file)
    reader = csv.DictReader(open(file_name, encoding="utf-8"))
    list_reader = list(reader)
    shuffle(list_reader)
    print("Sentiment:" , list_reader[0]['sentiment'])
    print("Review:", list_reader[0]['review'])

    total_samples = len(list_reader)
    test_size = int(total_samples *20/100)

    test_dataset = list_reader[:test_size]
    X_test = [e['review'] for e in test_dataset]
    y_test = [e['sentiment'] for e in test_dataset]

    train_dataset = list_reader[test_size:]
    X_train = [e['review'] for e in train_dataset]
    y_train = [e['sentiment'] for e in train_dataset]

    print("Train set size:", len(X_train), " , ", len(y_train))
    print("Test set size:", len(X_test), " , ", len(y_test))

    (X_train_encoded, X_test_encoded, vocab_size, max_length) = data_encoding(X_train, X_test)

    sentiments = {'negative': 0, 'positive': 1}
    y_train = [sentiments[item] for item in y_train]
    y_test = [sentiments[item] for item in y_test]

    y_train = array(y_train)
    y_test = array(y_test)

    # save the reshaped photos
    train_data_file_name = get_path(in_config.data_path_root, in_config.train_data_name)
    train_labels_file_name = get_path(in_config.data_path_root, in_config.train_labels_name)
    test_data_file_name = get_path(in_config.data_path_root, in_config.test_data_name)
    test_labels_file_name = get_path(in_config.data_path_root, in_config.test_labels_name)

    save(train_data_file_name, X_train_encoded)
    save(train_labels_file_name, y_train)
    save(test_data_file_name, X_test_encoded)
    save(test_labels_file_name, y_test)

    info_file_name = get_path(in_config.data_path_root, in_config.info_file_name)
    file = open(info_file_name, "w")
    file.writelines(str(vocab_size))
    file.writelines("\n")
    file.writelines(str(max_length))
    file.close()

    print("Data preparation completed!")

