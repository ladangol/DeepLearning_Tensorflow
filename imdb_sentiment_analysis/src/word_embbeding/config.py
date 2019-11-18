model_path_root = 'models'
data_path_root = 'data'

#training config
num_epochs = 2
batch_size = 32
add_gru_layer = True


# data prepration config
data_file = "IMDB Dataset.csv"
train_data_name = 'train_data.npy'
train_labels_name = 'train_labels.npy'
test_data_name = 'test_data.npy'
test_labels_name = 'test_labels.npy'
info_file_name = "info.txt"

# word2vec config
# Set values for various parameters
num_features = 10  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context_window = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words
word2vec_model = str(num_features)+"features_"+str(min_word_count)+"minwords_"+str(context_window)+"context"


