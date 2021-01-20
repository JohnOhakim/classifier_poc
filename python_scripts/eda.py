import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup 
import spacy
nlp = spacy.load('en')


train_df = pd.read_csv('./data/train.csv')

# def split_data(train_df, test_size=test_size, random_state=random_state):

#     X_train, X_test, y_train, y_test = train_test_split(train_df['posts_combined'],
#                                                     train_df['is_tv'],
#                                                     test_size=test_size,
#                                                     random_state=random_state)
#     return X_train, X_test, y_train, y_test
                                                
def post_to_words(raw_post, regex_s="[^a-zA-Z]"):

    post_text = BeautifulSoup(raw_post).get_text()
    letters_only = re.sub(regex_s, " ", post_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]

    return (" ".join(meaningful_words))   

train_df['clean_text'] = train_df['posts_combined'].apply(post_to_words)

def stem_text(text):    
    stm = PorterStemmer()
    tokenized = word_tokenize(text)
    stem_tokes = []
    for toke in tokenized:
        stem_tokes.append(stm.stem(toke))
    stem_tokes

    combined = ''
    for stemmed in stem_tokes:
        combined += stemmed + ' '
    return combined

def lem_text(text, regex_toks=r'[A-Z]\w+'):    
    lem = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(regex_toks)
    tokenized = tokenizer.tokenize(text)
    lemmed_tokes = []
    for toke in tokenized:
        lemmed_tokes.append(lem.lemmatize(toke))
    lemmed_tokes

    combined = ''
    for lemmed in lemmed_tokes:
        combined += lemmed + ' '
    return combined

train_df['stemmed_text'] = train_df['posts_combined'].apply(stem_text)
train_df['lemmed_text'] = train_df['posts_combined'].apply(lem_text)

def extract_PROPN(text):
    document = nlp(text)
    PROPN = []
    for token in document:
        if token.pos_ == 'PROPN':
            PROPN.append(str(token))   
    #print(PROPN)
    combined = ''
    for token in PROPN:
        combined += token + ' '
    return combined

train_df['pnouns_from_lemmed'] = train_df['lemmed_text'].apply(extract_PROPN)

def extractor(data):
    tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b') # Regex here says to get full words but exclude digits
    dirty_text_tokens = [tokenizer.tokenize(token.lower().strip()) for token in data]

    master_text_tokens = []

    for text in dirty_text_tokens:
        clean_text_tokens = []
        for word in text:
            if ((word not in stopwords.words('english')) and (word not in ['http','https','www','com','@','...','…'])):
                clean_text_tokens.append(word)
        master_text_tokens.append(clean_text_tokens)
        
    return master_text_tokens 


full_list = extractor(train_df['posts_combined'])
pnouns_list = extractor(train_df['pnouns_from_lemmed'])


def get_common_words(list_of_words, n):

    word_full_list = list_of_words[0]

    for sub_list in list_of_words[1:]:
        word_full_list.extend(sub_list)
    fdist_n = nltk.FreqDist(word_full_list)
    common_words = fdist_n.most_common(n)
    word_freq = []
    for word, freq in common_words:
        #print(word)
        full_common_words = {}
        full_common_words['words'] = word
        full_common_words['freq'] = freq
        word_freq.append(full_common_words)
    return pd.DataFrame(word_freq)

common_words = get_common_words(full_list, 20)
common_words_pnouns = get_common_words(pnouns_list, 20)

def plotter(x, y, part_of_speech, title_fontsize, label_fontsize):
    plt.figure(figsize=(10, 7))
    sns.barplot(x,
                y,  
                orient='h', color='b')
    plt.title(f'20 Most Common Words ({part_of_speech})', fontsize=title_fontsize)
    plt.xlabel('Frequency', fontsize=label_fontsize)
    plt.ylabel('Words', fontsize=label_fontsize)

plotter(common_words['freq'], common_words['words'], 'Clean Text', 18, 16)

plotter(common_words_pnouns['freq'],common_words_pnouns['words'], 'Proper Nouns', 18, 16)



train_df.to_csv('./data/train_clean.csv', index=False)