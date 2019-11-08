import pickle
import pandas as pd

from src.util import clean_text
test_phrases = {
    "Well done!": 1,
    "horrible acting": 0,
    "Good work": 1,
    "It was very boring ": 0,
    "I will not recommend": 0,
    "Its a fantastic series":1
}

secret_test_phrases = {
"pathetic picture": 1
}


def convertTuple(tup):
    str =  ''.join(tup)
    return str

def sentiment_analysis_result_Original(input_tweet, *args, **kwargs):
    # TODO: returm sentiment analysis on input tweet
    # 1: positive, 0 negative
    modelfilename = convertTuple(args)
    Model = pickle.load(open(modelfilename, "rb"))
    abc_series = pd.Series(input_tweet)
    result = Model.predict(abc_series)
    return result[0]

def sentiment_analysis_result(input_review, *args, **kwargs):
    # TODO: returm sentiment analysis on input tweet
    # 1: positive, 0 negative
    modelfilename, bowTransformefilename, TfidfTransformefilenam = args

    Model = pickle.load(open(modelfilename, "rb"))
    bow_transformer = pickle.load(open(bowTransformefilename, "rb"))
    tfidf_transformer = pickle.load(open(TfidfTransformefilenam, "rb"))

    input_review = clean_text(input_review)
    input_review = [input_review]  #convert it to the list because
    #transform needs a iterable object
    messages_bow_test = bow_transformer.transform(input_review)
    messages_transformer_test = tfidf_transformer.transform(messages_bow_test)
    result = Model.predict(messages_transformer_test)
    return result[0]


def validation(modelfilename, bowTransformefilename,TfidfTransformefilename):
    nbr_test_phrases = len(test_phrases)
    nbr_secret_test_phrases = len(secret_test_phrases)

    count_correct = 0
    for phrase, score in test_phrases.items():
        result = sentiment_analysis_result(phrase, modelfilename, bowTransformefilename,TfidfTransformefilename)
        print("phrase:  " + phrase )
        print("Model : " + str(result) + ", score: " + str(score))
        if result == score:
            count_correct += 1

    test_phases_count_correct = count_correct
    test_phrases_result = test_phases_count_correct/nbr_test_phrases * 100
    print('You have a score of {}% on the test phrases'.format(test_phrases_result))

    if test_phrases_result < 70:
    	print("You do not get a good score on the test phrases! {}".format("*" * 30))

    count_correct = 0
    for phrase, score in secret_test_phrases.items():
        result = sentiment_analysis_result(phrase, modelfilename, bowTransformefilename,TfidfTransformefilename)
        print("phrase:  " + phrase )
        print("Model : " + str(result) + ", score: " + str(score))


        if result == score:
            count_correct += 1

    secret_test_phases_count_correct = count_correct
    secret_test_phrases_result = 0.0
    if nbr_secret_test_phrases != 0:
    	secret_test_phrases_result = secret_test_phases_count_correct/nbr_secret_test_phrases * 100
    print('You have a score of {}% on the secret test phrases'.format(secret_test_phrases_result))

    total_test_phrases_result = ((test_phases_count_correct + secret_test_phases_count_correct)/ \
    	(nbr_test_phrases + nbr_secret_test_phrases)) * 100

    print('You have a total score of {}%'.format(total_test_phrases_result))
