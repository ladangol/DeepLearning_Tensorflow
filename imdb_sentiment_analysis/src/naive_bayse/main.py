import pandas as pd
from tester import validation, sentiment_analysis_result
from train import partial_train, partial_retrain, train

# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
# import string
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import re
# from nltk.tokenize import TweetTokenizer
# from sklearn.metrics import classification_report
# import datetime
# import pickle
# import csv

def User_interaction(modelfilename, bowTransformefilename,TfidfTransformefilename, traindatafilename):
    out_data = pd.DataFrame()

    messages = pd.read_csv('./data/Twitter/twitter-sentiment-analysis2/test.csv', encoding='ISO-8859-1',
                           engine='python')
    for i in range(0, len(messages)):
        phrase = messages.iloc[i].SentimentText
        result = sentiment_analysis_result(phrase, modelfilename, bowTransformefilename,TfidfTransformefilename)
        print("phrase:  " + phrase )
        if(result == 1):
            print("Model result: Positive")
        else:
            print("Model result: Negative/Natural")
        answer = input()
        print("Options: [p]Positive, [n]Negative/Natural, [q]uit \n Enter selection:")
        user_feedback = -1
        while True:
            answer = input()
            if(answer == "\n" or answer == ""):
                continue
            elif (answer.lower() == "p"):
                print("User enter positive!")
                user_feedback = 1
                break
            elif (answer.lower() == "n"):
                user_feedback = 0
                print("User enter negative or natural!")
                break
            elif (answer.lower() == "q"):
                print("Applicatin end by user!")
                out_data.to_csv(traindatafilename)
                return
            else:
                print("Please enter the valid option!")

        if result != user_feedback:
           out_data = out_data.append({'Sentiment': user_feedback, 'SentimentText': phrase}, ignore_index=True)

    out_data.to_csv(traindatafilename)

def main():
    while True:
        print("Options: [1]Training, [2]Validation, [3]User interaction, [4]Re-train ,[5]bouns,  [0]Quit \n Enter selection:")
        answer = input()
        if (answer == "\n" or answer == ""):
            continue
        if(answer == "0"):
            print("Applicatin end by user!")
            exit(0)
        elif(answer == "1"):
            partial_train('./data/Twitter/twitter-sentiment-analysis2/train.csv')
        elif(answer == "2"):
            validation("./Model/lastModel.p", "./Model/lastBowTransforme.p", "./Model/lastTfidfTransforme.p")
        elif (answer == "3"):
            User_interaction("./Model/lastModel.p","./Model/lastBowTransforme.p", "./Model/lastTfidfTransforme.p", "./Data/retrain.csv")
        elif (answer == "4"):
            partial_retrain("./Model/lastModel.p", "./Model/lastBowTransforme.p","./Model/lastTfidfTransforme.p",'./data/retrain.csv')
        elif(answer == "5"):
            train("./data/bonus.csv")
        else:
            print("Please enter the valid option!")

    return 0

if __name__ == "__main__":
    main()

