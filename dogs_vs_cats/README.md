
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

![image](https://drive.google.com/uc?export=view&id=1VkmH1G-GUjrrLo-paKoHLizPB0bMEyUz)

![image](https://drive.google.com/uc?export=view&id=1wA82jPTSyXspneTEnAj1x4vFE6A2-Rhx)

![image](https://drive.google.com/uc?export=view&id=1LnU0ur61JAX8GJGdeKJrILSr_0jAbn8Q)

## Class Activation Map
Using keras pre-trained model of VGG_16.

![image](https://drive.google.com/uc?export=view&id=1cB8-BNXWNnxnjvaB4OJ0NM7gJbS2V7KT)

![image](https://drive.google.com/uc?export=view&id=1U7c34ml46s2YhSlqegZSKv8Zm-qtRpg_)

- Example 1:

![image](https://drive.google.com/uc?export=view&id=1_ZMJctd4_nDW2M8jASW65mYGK21P1CuN)

- Example 2:

![image](https://drive.google.com/uc?export=view&id=1m2VgXXv1kpQkgKvkn528Fa6p6-9oeG0x)

- Example 3:

![image](https://drive.google.com/uc?export=view&id=1JZ7Br_OwxocrslV7OIBLvHcBKfh7W7vB)

- Example 4:

![image](https://drive.google.com/uc?export=view&id=1WNMYsRv7yVgzljjQZ74jx59-t7LFkDTb)

## Hyper Parameter Tuning:

In order to increase the accuracy of our model, we did a grid search on some of the parameters such as learning rate, and kernel initializer functions.

We also introduced a learning rate scheduler that as the epoch increases, slows down the learning rate so the model converges. Our initial experiments showed that with a learning rate of 0.1 the model diverges and do not learn anything as shown below:
with the learning rates [0.1(orange), 0.01(dark blue), 0.001(red), 0001(light blue)]

Training logs of grid serach on 4 learning rates:

![image](https://drive.google.com/uc?export=view&id=1njWcwqkRdboYoP5wXFWlywuxQabsIAcD)

The classification results of learning rate 0.1:

![image](https://drive.google.com/uc?export=view&id=1nNSliJavghFogLll_57UOO9QO3GUUGqk)

Training logs after removing the learning rate 0.1:

![image](https://drive.google.com/uc?export=view&id=1abYjVekvOh8jmn97q2m3orOGgd3vTypV)

In the next experiment we consider the initial learning rates of [0.01, 0.001]. After the epoch ?, the learning rate scheduler reduces the learning rate by 0.1 every 5 epochs.

The other hyper parameters that we investigate:

Learning rates: [0.01, 0.001]

Activation function: [swish, tanh, leakyrelu, relu]

kernel initializer: [he_uniform, glorot_uniform, lecun_uniform]

Bias initializer: [0.0, 0.01]

Training logs of grid serach:

![image](https://drive.google.com/uc?export=view&id=16Q-aBkie1JWs2jyc1Pc91uxnJYpVDtJU)

Worst 5 results: 

|HP Search	|Accuracy	|Precision	|Recall	|F1-Score|
|------ | --------|------ | --------|------ | 
|LR_0.01-KI_he_uniform-AC_Tanh-BI_0.01	|0.5	|0.25	|0.5|	0.33 |
|LR_0.01-KI_lecun_uniform-AC_Tanh-BI_0.0|	0.5|	0.25|	0.5	| 0.33 |
|LR_0.01-KI_he_uniform-AC_Tanh-BI_0.0	|0.5	|0.25|	0.5|	0.33 |
|LR_0.01-KI_he_uniform-AC_LeakyReLU-BI_0.01|	0.5	|0.25	|0.5 |	0.33 |
|LR_0.01-KI_he_uniform-AC_LeakyReLU-BI_0.0	|0.5	|0.25 |	0.5 |	0.33 |

Top 5 results:

|HP Search	|Accuracy	|Precision	|Recall	|F1-Score|
|------ | --------|------ | --------|------ | 
|LR_0.01-KI_glorot_uniform-AC_LeakyReLU-BI_0.01|	0.87 |	0.87	|0.87|	0.87|
|LR_0.01-KI_he_uniform-AC_ReLU-BI_0.01|	0.88 |	0.88|	0.88|	0.88|
|LR_0.01-KI_lecun_uniform-AC_ReLU-BI_0.0|	0.88	|0.88	|0.88	|0.88|
|LR_0.01-KI_lecun_uniform-AC_ReLU-BI_0.01|	0.89|	0.89	|0.89|	0.89|
|LR_0.01-KI_he_uniform-AC_ReLU-BI_0.0	|0.89	|0.89	|0.89	|0.89|

LR: learning rate, 
KI: kernel initializer, 
AC: Activation function, 
BI: Bias initializer
