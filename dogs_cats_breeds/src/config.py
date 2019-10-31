image_size = 224
num_epochs = 150
batch_size = 32
num_channels = 3

dict_categories = {0: 'Dog', 1: 'Cat'}

num_types = 2
num_breeds = 2

model_path_root = 'models'
data_path_root = 'data'

data_name = 'simple_dogs_vs_cats_photos_small.npy'
labels_name = 'simple_dogs_vs_cats_labels_small.npy'
confusion_matrix_file_name = "confusion_matrix.txt"
confusion_matrix_plot_name = "confusion_matrix.png"


EPOCHS = 1
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (image_size, image_size, num_channels)