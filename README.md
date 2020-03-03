
# DeepLearning_Tensorflow
This repo is for practicing deep learning algorithms using Tensorflow and Keras. 

## Tutorials
If you are not familiar with deep learning frameworks, I recommend to read [deep learning frameworks](doc/deep_learning_frameworks.md) document first.

## Installation
If you are using anaconda you install all the require library by importing the [tensorflow-v1-gpu.yaml](installation/tensorflow-v1-gpu.yaml).
Before importing the yml file please replace [username] by the valid name.
Then open anaconda prompt and enter the following comments
conda env create -f DeepLearning_Tensorflow/installation/environment.yml
if you do not have anaconda install the you need to install following libraries:
- Python 3.7+ is required
- tensorflow-gpu==1.14.0
- numpy
- opencv
- matplotlib
- pandas
- sklearn

## Project details
- **dogs_vs_cats:** Dogs vs. Cats classification project using keras and tensorflow. To monitor training tensorboard is used. The model can be learned from scratch or using transfer learning. Class activation maps are implemented to visualize the focus of  the convolutional layers. 
- **imdb_sentiment_analysis:** is a simple chatbot application based on naive bayes bag of words representation of words.  
- **input_pipeline:** building an data pipeline using tensorflow tf.data to classify images of cats and dogs.
