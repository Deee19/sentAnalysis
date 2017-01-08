from sklearn.externals import joblib
from sklearn import metrics
from time import time
import nltk
import re


class Preprocess():

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
            ]

    def tokenize(self, document):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(document) for word in nltk.word_tokenize(sent)]
        # Break the sentence into part of speech tagged tokens
        filtered_tokens = []
        for token in tokens:
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            if re.search('[a‐zA‐Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens
class classify():

    def feature_extraction_classify(words):

        # # loading trained model
        outpath = r'C:\Users\Dharmie\PycharmProjects\sentAnalysis\trainedmodel\svm_classify.pkl'
        svc_model = joblib.load(outpath)


        #Make predictions
        # pred_test = tfidf.transform([words])
        predicted = svc_model.predict(words)
        return predicted
