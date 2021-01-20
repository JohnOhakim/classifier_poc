import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import re


holdout = pd.read_csv('./data/test.csv')

def post_to_words(raw_post, regex_s="[^a-zA-Z]"):

    post_text = BeautifulSoup(raw_post).get_text()
    letters_only = re.sub(regex_s, " ", post_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))   

holdout['clean_text'] = holdout['posts_combined'].apply(post_to_words)

X_hold = holdout['posts_combined']
y_hold = holdout['is_tv']

def text_parser(holdout_set):

    total_post = hold_set.shape[0]
    print(f'There are {total_post} posts.')

    clean_X_hold = []

    print("Cleaning and parsing the holdout set movie/tv reviews...")

    j = 0
    for docs in hold_set:
        clean_X_hold.append(post_to_words(docs))
        if (j + 1) % 250 == 0:
            print(f'Post {j + 1} of {total_post}.')
        j += 1
    return pd.Series(clean_X_hold)

X_hold_clean = text_parser(X_hold)    

train = pd.read_csv('./data/train_clean.csv')

