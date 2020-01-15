image_size = 224
num_epochs = 150
batch_size = 32

dict_categories = {0: 'Dog', 1: 'Cat'}

num_classes = 2

model_path_root = 'models'
data_path_root = 'data'

data_name = 'simple_dogs_vs_cats_photos.npy'
labels_name = 'simple_dogs_vs_cats_labels.npy'
confusion_matrix_file_name = "confusion_matrix.txt"
confusion_matrix_plot_name = "confusion_matrix.png"
# Test1:
# initial_lrate = 0.0001
# Test2:
# initial_lrate = 0.001
# Test3:
initial_lrate = 0.01