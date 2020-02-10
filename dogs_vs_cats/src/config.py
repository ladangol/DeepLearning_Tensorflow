#General
model_path_root = 'models'
data_path_root = 'data'

#Data
x_train_path='x_train_small.npy'
y_train_path='y_train_small.npy'
x_test_path='x_test_small.npy'
y_test_path='y_test_small.npy'

dict_categories = {0: 'Dog', 1: 'Cat'}
small_data_set_size = 0


#model
activation = 'ReLU'
kernel_initializer = 'he_uniform'
bias_initializer = 0.0
#for inception-resnet set the image_size to 87 otherwise 224
image_size = 224
num_channel = 3
num_classes = 2

#Training
num_epochs = 2
batch_size = 5
initial_lrate = 0.0001

#evaluation
confusion_matrix_detailed_file_name = "confusion_matrix_detailed.txt"
confusion_matrix_file_name = "confusion_matrix.txt"
confusion_matrix_plot_name = "confusion_matrix.png"
display_plot = True
grid_serach_validation_result = {}

