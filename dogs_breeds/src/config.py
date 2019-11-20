image_size = 224
num_epochs = 150
batch_size = 32
save_model = False
debug_image = False

dict_categories = {}

num_classes = -1
num_channels = 3

model_path_root = 'models'
data_path_root = 'data'

data_name = 'dogs_breeds_photos.npy'
labels_name = 'dogs_breeds_labels.npy'
lable_index_map = "labels.json"
breeds_distribution = 'breeds_distribution.json'

confusion_matrix_file_name = "confusion_matrix.txt"
confusion_matrix_plot_name = "confusion_matrix.png"