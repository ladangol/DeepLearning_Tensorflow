# Dogs_breeds
A sample project to classify images of dogs breeds 

## Folder Structure
- dogs_breeds
  - src
    - main.py
    - data_prepration.py
    - config.py
    - util.py
  - data
    - train
      - *Annotations for your images will come here*
      - *All of your images will come here* 
    - test
        - *All your images for testing will come here*	  
    - data_prepration.py will use the train images along with their labels to generate the following files:
        - breeds_distribution.json
        - dogs_breeds_labels.npy
        - dogs_breeds_photos.npy
        - labels.json  
   - models
     - *Inference graph of the trained model will be saved here*
   - doc
     - Useful tutorials and links can be found here

## Code Explanation
- config.py: Define global variables.
- data_prepration.py: Preprocess the train data for training.
- main.py: Depend on the user input, run the training or the prediction function.
- util.py: Contains utility functions used during training or prediction.

## How to run the code
python DeepLearning_Tensorflow/dogs_breeds/src/main.py 

Checkpoints will be saved in DeepLearning_Tensorflow/dogs_breeds/model folder. 

## Training results
