import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
import datetime
import pickle
from src.util import  clean_text

def train(datafilename):
    print("Start starts training, this might take a while ....")
    messages = pd.read_csv(datafilename, encoding='ISO-8859-1', engine='python')
    msg_train, msg_test, label_train, label_test = train_test_split(messages['tweet'], messages['res'], test_size = 0.2)

    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=clean_text)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    model = pipeline.fit(msg_train, label_train)
    filename = "./Model/./BonusModel_" + str(datetime.datetime.now()) + ".p"
    filename = filename.replace(':', '-')
    filename = filename.replace(' ', '_')
    pickle.dump(model, open(filename,"wb"))
    filename = "./Model/lastBonusModel.p"
    pickle.dump(model, open(filename, "wb"))

    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions, label_test))


def partial_train(datafilename):
    print("Start training, this might take a while ....")
    messages = pd.read_csv(datafilename, encoding='ISO-8859-1', engine='python')
    msg_train, msg_test, label_train, label_test = train_test_split(messages['SentimentText'], messages['Sentiment'],
                                                                    test_size=0.2)

    # check later if you need to use this on the whole corpus or just train_set
    bow_transformer = CountVectorizer(analyzer=clean_text).fit(messages['SentimentText'])
    messages_bow = bow_transformer.transform(msg_train)
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    model = MultinomialNB()
    model.partial_fit(messages_tfidf, label_train, classes=np.unique(label_train))
    #Saving model
    date_str = str(datetime.datetime.now())
    date_str = date_str.replace(':', '-')
    date_str = date_str.replace(' ', '_')

    filename = "./Model/Model_" + date_str + ".p"
    pickle.dump(model, open(filename,"wb"))
    filename = "./Model/lastModel.p"
    pickle.dump(model, open(filename, "wb"))

    filename = "./Model/BowTransforme_" + date_str + ".p"
    pickle.dump(bow_transformer, open(filename, "wb"))
    filename = "./Model/lastBowTransforme.p"
    pickle.dump(bow_transformer, open(filename, "wb"))

    filename = "./Model/TfidfTransforme_" + date_str + ".p"
    pickle.dump(tfidf_transformer, open(filename, "wb"))
    filename = "./Model/lastTfidfTransforme.p"
    pickle.dump(tfidf_transformer, open(filename, "wb"))

    # type(model)
    # no need for fit here , because it is already fitted with the msg_train
    messages_bow_test = bow_transformer.transform(msg_test)
    messages_transformer_test = tfidf_transformer.transform(messages_bow_test)
    # messages_tfidf_test = tfidf_transformer_test.transform(messages_bow_test)
    predictions = model.predict(messages_transformer_test)

    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    print(classification_report(predictions, label_test))
    print(accuracy_score(label_test, predictions))


def partial_retrain(modelfilename, bowTransformefilename,TfidfTransformefilename, datafilename):
    print("starts training, this might take a while ....")
    messages = pd.read_csv(datafilename, encoding='ISO-8859-1', engine='python')
   # msg_train, msg_test, label_train, label_test = train_test_split(messages['SentimentText'], messages['Sentiment'], test_size = 0.2)
    msg_train = messages['SentimentText']
    label_train = messages['Sentiment']

    #check later if you need to use this on the whole corpus or just train_set
    # bow_transformer = CountVectorizer(analyzer=clean_tweets).fit(messages['SentimentText'])
    # messages_bow = bow_transformer.transform(msg_train)
    # tfidf_transformer = TfidfTransformer().fit(messages_bow)
    # messages_tfidf = tfidf_transformer.transform(messages_bow)

    bow_transformer = pickle.load(open(bowTransformefilename, "rb"))
    tfidf_transformer = pickle.load(open(TfidfTransformefilename, "rb"))

    messages_bow = bow_transformer.transform(msg_train)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    model = pickle.load(open(modelfilename, "rb"))
    model.partial_fit(messages_tfidf, label_train, classes=np.array([0,1]))

    #Saving model
    date_str = str(datetime.datetime.now())
    date_str = date_str.replace(':', '-')
    date_str = date_str.replace(' ', '_')

    filename = "./Model/Model_" + date_str + ".p"
    pickle.dump(model, open(filename,"wb"))
    filename = "./Model/lastModel.p"
    pickle.dump(model, open(filename, "wb"))

    # filename = "./Model/BowTransforme_" + date_str + ".p"
    # pickle.dump(bow_transformer, open(filename, "wb"))
    # filename = "./Model/lastBowTransforme.p"
    # pickle.dump(bow_transformer, open(filename, "wb"))
    #
    # filename = "./Model/TfidfTransforme_" + date_str + ".p"
    # pickle.dump(tfidf_transformer, open(filename, "wb"))
    # filename = "./Model/lastTfidfTransforme.p"
    # pickle.dump(tfidf_transformer, open(filename, "wb"))
    # print("Re-training is completed!")
    return
    #type(model)
    #no need for fit here , because it is already fitted with the msg_train
    messages_bow_test = bow_transformer.transform(msg_test)
    messages_transformer_test = tfidf_transformer.transform(messages_bow_test)
    #messages_tfidf_test = tfidf_transformer_test.transform(messages_bow_test)
    predictions = model.predict(messages_transformer_test)


    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    print (classification_report(predictions,label_test))
    print(accuracy_score(label_test, predictions))
