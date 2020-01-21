import os
import matplotlib.pylab as plt
import itertools
import numpy as np

def get_path(in_root, in_folder):
    return os.path.join(in_root, in_folder)

def get_category(in_config, in_key):
    return in_config.dict_categories.get(in_key, None)

def get_index(in_config, in_value):
    if not isinstance(in_value, str):
        return -1

    for key, val in in_config.dict_categories:
        if val == in_value:
            return key
    return -1

def get_index_by_image_name(in_config, in_file_name):
    for key, val in in_config.dict_categories:
        if in_file_name.lower().startswith(val.lower()):
            return key
    return -1


def get_categories(in_config):
    return in_config.dict_categories.values()

def plot_confusion_matrix(in_confusion_matrix, in_config, normalize=False):
    classes = list(in_config.dict_categories.values())

    fig = plt.figure(figsize = (5,5))
    plt.imshow(in_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        in_confusion_matrix = in_confusion_matrix.astype('float') / in_confusion_matrix.sum(axis=1)[:, np.newaxis]

    thresh = in_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(in_confusion_matrix.shape[0]), range(in_confusion_matrix.shape[1])):
        y=0.7
        if i==0:
            y=0.2
        plt.text(j, y, in_confusion_matrix[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if in_confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    confusion_matrix_plot_name = get_path(get_path('Models','logs'), in_config.confusion_matrix_plot_name)
    plt.savefig(confusion_matrix_plot_name)

    if in_config.display_plot:
        plt.show()
    plt.close(fig)

    confusion_matrix_file_name = get_path(get_path('Models','logs'), in_config.confusion_matrix_file_name)
    with open(confusion_matrix_file_name, 'w') as f:
        f.write(np.array2string(in_confusion_matrix, separator=', '))


def generate_current_config_to_string(in_config):
        return "-E_"   + str(in_config.num_epochs) + \
              "-LR_" + str(in_config.initial_lrate) + \
              "-KI_" + str(in_config.kernel_initializer) + \
              "-AC_" + str(in_config.activation) + \
              "-BI_" + str(in_config.bias_initializer) + \
              "-"









