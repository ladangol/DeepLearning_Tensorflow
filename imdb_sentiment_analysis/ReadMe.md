# imdb_sentiment_analysis
A sample project to classify images of dogs and cats and dogs breeds. 

## Folder Structure
- imdb_sentiment_analysis
  - src
    - util.py
    - naive_bayse
      - main.py
      - tester.py
      - train.py
    - word_embbeding
      - main.py
      - build_model.py
      - build_word2vec_model.py
      - data_prepration.py
      - train.py
      - config.py

  - data
     - data_prepration.py aftre cleaning and tokenizing the data of IMDB Dataset will generate the following files:
        - test_data.npy
        - test_labels.npy
        - train_data.npy
        - train_labels.npy      
   - models
     - *Inference graph of the trained model will be saved here*
   - doc
     - Useful tutorials and links can be found here

## Code Explanation
- util.py
- naive_bayse
    - main.py
    - tester.py
    - train.py
 - word_embbeding
    - main.py
    - build_model.py
    - build_word2vec_model.py
    - data_prepration.py
    - train.py
   - config.py: Define global variables.
  
## How to run the code
python DeepLearning_Tensorflow/imdb_sentiment_analysis/srcnaive_bayse/main.py 
python DeepLearning_Tensorflow/imdb_sentiment_analysis/word_embbeding/main.py 

Checkpoints will be saved in DeepLearning_Tensorflow/imdb_sentiment_analysis/model folder. 

## Training results
