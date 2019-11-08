from  build_model import  define_model
from src.util import  get_path
from numpy import  load

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import keras

def train(in_config):
    print("Loading data!")

    test_data_path = get_path(in_config.data_path_root, in_config.test_data_name)
    test_labels_path = get_path(in_config.data_path_root, in_config.test_labels_name)
    test_data = load(test_data_path)
    test_labels = load(test_labels_path)

    train_data_path = get_path(in_config.data_path_root, in_config.train_data_name)
    train_labels_path = get_path(in_config.data_path_root, in_config.train_labels_name)
    train_data = load(train_data_path)
    train_labels = load(train_labels_path)
    model = define_model(in_config)
    # fit the model
    model.fit(train_data, train_labels, batch_size=in_config.batch_size, epochs=in_config.num_epochs, verbose=1)
    # evaluate the model
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print('Validation accuracy: %f' % (accuracy * 100))

    print('saving model')
    model_name = get_path(in_config.model_path_root, "imdb_sentiment_analysis")
    model_full_path = "{0}-epoch-{1:02d}-epoch_loss-{2:.3f}-accuracy-{2:.3f}".format(model_name, in_config.num_epochs + 1, loss, accuracy)
    model.save(model_full_path)

def predict(in_config):
    test_sample_1 = "This movie is fantastic! I really like it becuase it os so good!"
    test_sample_2 = "Good movie!"
    test_sample_3 = "Maybe I like this movie."
    test_sample_4 = "Not to my taste, will skip and watch another movie"
    test_sample_5 = "if you like action, then this movie might be good for you"
    test_sample_6 = "bad movie!"
    test_sample_7 = "Not a good movie!"
    test_sample_8 = "This movie realy sucks, Can I get my money back please"
    test_samples = [test_sample_1,
                    test_sample_2,
                    test_sample_3,
                    test_sample_4,
                    test_sample_5,
                    test_sample_6,
                    test_sample_7,
                    test_sample_8]

    max_length = max([len(s.split()) for s in test_samples])
    tokenizer_obj = Tokenizer()
    test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

    model_name = get_path(in_config.model_path_root, "imdb_sentiment_analysis")
    model = keras.models.load_model(model_name)
    prediction = model.predict(test_samples_tokens_pad)
    print(prediction)