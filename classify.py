import re
import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  Normalizer
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

def feature_extraction_cluster(words):

    t0 = time()
    print (words)
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=1000, min_df=2, tokenizer=tokenize_only,
                            ngram_range=(1, 1), use_idf=True)
    print(type(words))
    tfs = tfidf.fit_transform(words)
    print("done in %fs" % (time() - t0))
    term = tfidf.get_feature_names()
    print(term)
    print (tfs.shape)

#Read the tweets one by one and process it

inpTweets = pd.read_excel(r"C:\Users\Dharmie\PycharmProjects\sentAnalysis\Book1.xlsx")
Text =inpTweets['Tweets']
result = []
for line in Text:
    result.append(line)
# print(result)

feature_extraction_cluster(result)
