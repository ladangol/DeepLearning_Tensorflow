
# Dogs_vs_Cats
A sample project to classify images of cats and dogs 

## Folder Structure
- dogs_vs_cats
  - src
    - cam.py
    - simple_cnn.py
    - config.py
    - main.py
    - util.py
    - data_prepration.py
  - data
    - train
      - *Annotations for your images will come here*
      - *all of your images will come here* 
    - test
    - labels.npy 
    - data.npy

   - models
     - *inference graph of the trained model will be saved here*



	

## Setup to use dogs_vs_cats repository 



## Explore Different Training
Build a Custom Training:
If you want to build a custom training run simple_cnn.py loads all the dogs and cats images in memory without any image augmentaion.

Transfer Learning: 
If you do not have enough memory to load all the data in memory use XXX.py

If you are intersted in transfer learning run transfer_learning.py

## Code Explanation
- config.py: Define global variables.
- data_prepration.py: Preprocess the train data for training.
- main.py: Depend on the user input, run the training or the prediction function.
- util.py: Contains utility functions used during training or prediction.
- simple_cnn.py: Build a model from scratch. 
- cam.py: Build calss activation map.

## How to run the code

python DeepLearning_Tensorflow/dogs_vs_cats/src/main.py 
Checkpoints will be saved in  DeepLearning_Tensorflow/dogs_vs_cats/model folder. 

## Training results

The model is then fit and evaluated, which takes approximately 1 hour on modern GPU hardware.

Your specific results may differ given the stochastic nature of the learning algorithm.

In this case, we can see that the model achieved an accuracy of about 82% on the validation dataset as shown in the following figure.

Reviewing this figure, we can see that the model has overfit the training dataset at 60 epochs.

![image](../dogs_vs_cats/models/simple-cnn_acc_loss_graph_v1.jpg)
![image](../dogs_vs_cats/models/simple-cnn_confusion_matrix.jpg)
![image](../dogs_vs_cats/models/simple-cnn_network_evaluation.jpg)

## Class Activation Map
Using keras pre-trained model of VGG_16.

![image](../dogs_vs_cats/models/cam_vgg_16/confusion_matrix.jpg)
![image](../dogs_vs_cats/models/cam_vgg_16/network_evaluation.jpg)

- Example 1:

![image](../dogs_vs_cats/models/cam_vgg_16/cam_cat_2.jpg)

- Example 2:

![image](../dogs_vs_cats/models/cam_vgg_16/cam_cat_3.jpg)

- Example 3:

![image](../dogs_vs_cats/models/cam_vgg_16/cam_dog_3.jpg)

- Example 4:

![image](../dogs_vs_cats/models/cam_vgg_16/cam_dog_5.jpg)



