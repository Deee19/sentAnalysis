import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import metrics
from time import time
import os.path

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


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

def feature_extraction_classify(words):

    t0 = time()
    tfidf = TfidfVectorizer(stop_words='english', tokenizer= identity, preprocessor=None, lowercase=False )
    t0 = time()
    svc_model = SVC(kernel='linear')
    print("done in %fs" % (time() - t0))

    # pipeline
    pipeText = Pipeline([('preprocess', Preprocess()),
                         ('tfidf', tfidf),
                         ('svc_model', svc_model),

                         ])
    pipeText = pipeText.fit(X_train.values.astype('U'), y_train)

    # saving trained model
    outpath = r'C:\Users\Dharmie\PycharmProjects\sentAnalysis\trainedmodel\svm_classify.pkl'
    joblib.dump(pipeText, outpath)

#Read the tweets one by one and process it

inpTweets = pd.read_excel(r"Book1.xlsx")
X = inpTweets['Tweets']
y = inpTweets['cluster']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

feature_extraction_classify(inpTweets['Tweets'])







