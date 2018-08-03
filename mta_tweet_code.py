import pandas as pd
import numpy as np
import os
import re
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

stops = nltk.corpus.stopwords.words('english')
train_words = ['train', 'trains', 'st', 'av', 'http']
stops = stops+train_words

# lemmatizes 't' with 'lem'
def lemma(t, lem=None):
    if (lem is None) or (type(lem) != WordNetLemmatizer):
        lem = WordNetLemmatizer()
    t_lem = []
    for w in t.split():
        t_lem.append(lem.lemmatize(w))
    return ' '.join(t_lem)

# stems 't' with 'stem'
def stem(t, stem=None):
    if (stem is None):
        stem = PorterStemmer()
    t_stem = []
    for w in t.split():
        t_stem.append(stem.stem(w))
    return ' '.join(t_stem)

# processes (using 'pro') data['tweet_clean'] from 'start' to 'end'
def span_vec(start, end, data=df2, pro='stem'):
    tweets = data.set_index('date').loc[start:end]['tweet_clean']
    if pro == 'stem':
        t_pro = tweets.map(stem)
    elif pro == 'lem':
        t_pro = tweets.map(lemma)
    else:
        t_pro = tweets
    cv = CountVectorizer(stop_words=stops)
    v_tweets = cv.fit_transform(t_pro)
    gram = pd.DataFrame(v_tweets.todense(), columns=cv.get_feature_names()).sum().sort_values(ascending=False)
    return gram

# returns top 'n' words in data['tweet_clean'] per date range specified
def vecs_top_n(n=25, s='2017-01-01', freq='W', pds=83, data=df2, pro='stem'):
    d_range = pd.date_range(s, periods=pds, freq=freq)
    df = pd.DataFrame()
    start = d_range[0].strftime('%Y-%m-%d')
    for l in range(0, len(d_range)-1):
        if l % 5 == 0:
            clear_output()
        end = d_range[l+1].strftime('%Y-%m-%d')
        gram = span_vec(start, end, data=data, pro=pro)
        df_gram = pd.DataFrame(gram).reset_index().rename(columns={'index':'word',0:'count'}).iloc[0:n,:]
        df_gram['date'] = start
        df = pd.concat([df, df_gram])
        start = end
        print(start)
    return df

# plots usage of words 'i' thru 'j' in 'data'
def plot_words(i, j, data=top25):
    words = []
    for w in data.groupby('word').sum().sort_values('count', ascending=False).iloc[i:j].reset_index()['word']:
        words.append(w)

    plt.figure(figsize=(15,7))
    for w in words:
        plt.plot_date(data=data[data['word'] == w], y='count', x='date', xdate=True, fmt='-')
    plt.legend(labels=words)
    plt.ylim(0,1000)
    plt.xticks([]);