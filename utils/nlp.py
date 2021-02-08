'''
Utilities that process the text information from the reviews.
'''

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import config

def rm_punc_from_word(word):
    return ''.join([c for c in word if c not in string.punctuation])

stopwords_eng = stopwords.words('english')
stopwords_nopunc = set(map(rm_punc_from_word, stopwords.words('english')))

def clean_text(txt):
    '''
d    Remove punctuation and stop words from a single text.
    '''
    # Need to remove puntuation chars
    # txt.split() get things like 'food.\\n\\nI'
    # words = txt[2:-1].replace('\\n',' ') # txt = "b'...'"
    words = txt.replace('\\n',' ') # txt = "b'...'"    
    words = [c for c in words if c not in string.punctuation]
    words = ''.join(words)
    # # '->'.join(['Seattle', 'Boston', 'NYC']): 'Seattle->Boston->NYC'
    # # ''.join(['a','b','c',' ','e']): 'abc e'
    words = [wd for wd in words.split() if wd.lower() not in stopwords_nopunc]
    return ' '.join(words)

# txt=clean_text(txt)
# WordPunctTokenizer().tokenize(txt)

def item_item_matrix(df, max_features=100, max_df=0.8, min_df=1):
    '''
    Generate the item-item matrix based on consine similarity
    '''
    if('UID' and 'IID' in df): UID, IID = 'UID', 'IID'
    else: UID, IID = 'user_id', 'business_id'
    
    df['text'] = df['text'].apply(clean_text)

    # Now build a item-item content based recommender system
    # First, group by item (business ID)
    business_review = df.groupby(IID)['text'].apply(' '.join).reset_index()
    business_review.head()
    bids = business_review[IID]

    vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=max_features, max_df=max_df, min_df=min_df)
    # max_features picks features that have the highest term frequency across the corpus
    wordvec = vectorizer.fit_transform(business_review['text'])
    wordfeat = vectorizer.get_feature_names()
    business_matrix = cosine_similarity(wordvec)
    business_matrix = pd.DataFrame(business_matrix, columns=bids, index=bids)
    return business_matrix, wordvec, wordfeat

def show_feature_for_text(wordvec, column_index):
    '''
    Usage: show_feature_for_text(VectorizedDataFrame, column_index)
    Purpose: Show a list of non-zero TF-IDF values for some feature.
    '''
    row = wordvec.getrow(column_index)
    idx, val = row.indices, row.toarray()[0][row.indices]
    keyw = [wordfeat[i] for i in idx]
    data = pd.DataFrame({"keyword":keyw, "value":val})    
    data.sort_values(by="value", ascending=False)
    print(data)

if __name__ == "__main__":
    df = pd.read_csv(config.CSV_BUSINESS)
    df_review = pd.read_csv(config.CSV_REVIEW, nrows=20)
    # Check number of business_ids
    print("Unique Business ID: ", df_review.business_id.nunique())
    txt = df_review['text'][0]

# https://developers.google.com/machine-learning/recommendation/dnn/training
# https://towardsdatascience.com/yelp-restaurant-recommendation-system-capstone-project-264fe7a7dea1

