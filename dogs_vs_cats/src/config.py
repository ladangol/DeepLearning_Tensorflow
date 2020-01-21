#General
model_path_root = 'models'
data_path_root = 'data'

#Data
trainX_path='trainX.npy'
trainY_path='trainY.npy'
testX_path='testX.npy'
testY_path='testY.npy'

dict_categories = {0: 'Dog', 1: 'Cat'}

#model
activation='ReLU'
kernel_initializer = 'he_uniform'
bias_initializer = "zeros"
image_size = 224
num_classes = 2

#Training
num_epochs = 150
batch_size = 32
initial_lrate = 0.0001

#evaluation
confusion_matrix_detailed_file_name = "confusion_matrix_detailed.txt"
confusion_matrix_file_name = "confusion_matrix.txt"
confusion_matrix_plot_name = "confusion_matrix.png"
display_plot = True
grid_serach_validation_result = {}

