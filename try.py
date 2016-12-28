import re
import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  Normalizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from time import time

stemmer = WordNetLemmatizer()

def tokenize_and_stem(text):

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a‐zA‐Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.lemmatize(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):


    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a‐zA‐Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens




#Read the tweets one by one and process it

inpTweets = pd.read_excel(r"C:\Users\Dharmie\PycharmProjects\sentAnalysis\Book1.xlsx")
inpTweets.shape
X = inpTweets['Tweets']
y = inpTweets['cluster']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

vect = CountVectorizer()
hi = vect.fit_transform(X_train)
print(hi)

nb = MultinomialNB()
nb.fit(hi, y_train)
# make prediction
y_pred_classs = nb.predict(hi)
# calculate accuracy
metrics.accuracy_score(y_test, y_pred_classs)
# confusion matrix
metrics.confusion_matrix(y_test, y_pred_classs)
# transform data to a document-term matrix
# dami = vect.transform(words)
# print(dami)
# convert sparse matrix to dense matrix
# dami.toarray()
# examine vocabulary and document-term matrix together
# pd.DataFrame(dami.toarray(), columns=vect.get_feature_names())