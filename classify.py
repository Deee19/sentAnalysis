import re
import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
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

def feature_extraction_classify(words):

    t0 = time()
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=1000, min_df=2, tokenizer=tokenize_only,
                            ngram_range=(1, 1), use_idf=True)
    tfs = tfidf.fit_transform(X_train.values.astype('U'))
    print("done in %fs" % (time() - t0))
    # term = tfidf.get_feature_names()
    # print(term)
    # print (tfs.shape)

    # Classify
    nb = MultinomialNB().fit(tfs, y_train)

    # print(nb)

    #Make predictions
    pred_test = tfidf.transform(X_test.values.astype('U'))
    predicted = nb.predict(pred_test)

    # for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, y_test))

    # saving trained model
    # joblib.dump(nb, 'nb_classify.pkl')

    # raising valueerror
    print(nb.score(X_train, y_train))
    print(nb.score(X_test, y_test))
    # calculate accuracy
    print(metrics.accuracy_score(y_test, predicted))
    # confusion matrix
    print(metrics.confusion_matrix(y_test, predicted))


#Read the tweets one by one and process it

inpTweets = pd.read_excel(r"Book1.xlsx")
X = inpTweets['Tweets']
y = inpTweets['cluster']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction_classify(inpTweets['Tweets'])
