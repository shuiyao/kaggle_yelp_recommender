'''
Sentiment analysis on the Yelp review text using LSTM network.
'''

import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import tensorflow as tf
import string
import re
import shutil
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import config

import numpy as np

def rm_punc_from_word(word):
    return ''.join([c for c in word if c not in string.punctuation])

stopwords_eng = stopwords.words('english')
stopwords_nopunc = set(map(rm_punc_from_word, stopwords.words('english')))

def preprocess_text(txt):
    '''
    Pre-process text reviews before feeding them to TextVectorization()
    The TensorFlow TextVectorization() takes care of many pre-processing procedures such as removing punctuations and tokenizing the text. In the yelp reviews, we only need to remove the "\\n" characters.
    '''
    words = txt.replace('\\n',' ') # txt = "b'...'"    
    # words = [c for c in words if c not in string.punctuation]
    # words = ''.join(words)
    # words = [wd for wd in words.split() if wd.lower() not in stopwords_nopunc]
    # words = [wd.lower() for wd in words.split()]
    return words

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

__mode__ = "evaluate"

def get_glove_embeddings(vocab):
    '''
    Get pre-trained GloVe embeddings for the vocabulary from the training text.

    Parameters
    ----------
    vocab: <class 'set'>. The vocabulary from the training set.
    '''
    
    GLOVE_EMBEDDING = "/home/shuiyao/nlp_data/glove/glove.6B.50d.txt"

    PAD_TOKEN = 0

    word2idx = {'PAD': PAD_TOKEN}
    weights = []

    with open(GLOVE_EMBEDDING, 'r') as file:
        for index, line in enumerate(file):
            spt = line.split()
            word = spt[0]
            if(word in vocab):
                word_weights = np.asarray(spt[1:], dtype=np.float32)
                word2idx[word] = index + 1 # PAD is our zeroth index so shift by one
                weights.append(word_weights)

    EMBEDDING_DIMENSION = len(weights[0])
    # Insert the PAD weights at index 0 now we know the embedding dimension
    weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))

    # Append unknown and pad to end of vocab and initialize as random
    UNKNOWN_TOKEN=len(weights)
    word2idx['UNK'] = UNKNOWN_TOKEN
    weights.append(np.random.randn(EMBEDDING_DIMENSION))

    # Construct our final vocab
    weights = np.asarray(weights, dtype=np.float32)

    VOCAB_SIZE=weights.shape[0]
    # weights: (VOCAB_SIZE, 50)
    return word2idx, weights

def show_encoded_text(txt, encoder):
    vocab = np.array(encoder.get_vocabulary())
    encoded = encoder(txt).numpy()
    print(txt[i])
    print("----------------------------------------------------------------")
    print(" ".join(vocab[encoded]))

def show_example_for_first_n_text(dataset, encoder, n=1):
    vocab = np.array(encoder.get_vocabulary())
    for txt, label in dataset.take(1):
        encoded = encoder(txt).numpy()
        txt, label = txt.numpy(), label.numpy()
        for i in range(n):
            print(label[i])
            print(txt[i])
            print("--------------------------------")
            print(" ".join(vocab[encoded[i]]))

    # x = dataset.as_numpy_iterator()
    # y = x.next()
    # y is a tuple for the next batch/item (array['text'], array['target'])

MAX_VOCAB_SIZE = 5000
BUFFER_SIZE = 1000 # For dataset random shuffling
BATCH_SIZE = 128
TRAIN_SIZE = 0.8
NROWS = 20000

if __mode__ == "train":
    df_review = pd.read_csv(config.CSV_REVIEW, nrows=NROWS)
    data = df_review[['text','stars']]
    data['text'] = data['text'].apply(preprocess_text)    

    train_tail = (int)(NROWS * TRAIN_SIZE)
    train = data.loc[:train_tail-1]
    test = data.loc[train_tail:]
    dset = tf.data.Dataset.from_tensor_slices((train['text'], train['stars']))
    dset = dset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    tset = tf.data.Dataset.from_tensor_slices((test['text'], test['stars']))
    tset = tset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    encoder = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE
    ) # Good enough

    encoder.adapt(dset.map(lambda x, y : x))
    vocabset = set(encoder.get_vocabulary())
    vocab_size = len(encoder.get_vocabulary())

    show_example_for_first_n_text(dset, encoder, n=1)

    word2idx, weights = get_glove_embeddings(vocabset)

    mat = np.zeros((vocab_size, weights.shape[1]))
    for i, word in enumerate(encoder.get_vocabulary()):
        vec = word2idx.get('')
        if(vec is not None):
            mat[i] = weights[vec]

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = weights.shape[1],
            weights = mat, # from the super base layer super class
            # embeddings_initializer = tf.keras.initializers.Constant(mat),
            mask_zero=True,
            trainable=True
        ),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['RootMeanSquaredError'])

    # After 10 epochs, the RMSE drops below 1.0 for the regression task.

    history = model.fit(dset, epochs=10,
                        validation_data=tset,
                        validation_steps=30)

if __mode__ == "evaluate":
    test = ["Amazing!",
            "The food is amazing!",
            "The restaurant is amazing!",
            "Wow the restaurant is amazing!",
            "Wow wow wow the restaurant is amazing!",
            "Oh oh oh the restaurant is amazing!",
            "The restaurant is amazing! I really like it. Definitely recommend it."]
    pred = model.predict(test)
    print("Some Results: ")
    for i, txt in enumerate(test):
        print("{a:5.3f}: {b}".format(a=pred[i][0], b=txt))
    # history.history.keys() = dict_keys(['loss', 'root_mean_squared_error', 'val_loss', 'val_root_mean_squared_error'])
    
    test = train['text'][:200].to_list()
    score = train['stars'][:200].to_list()
    pred = model.predict(test)
    diff = np.abs(score - pred.flatten())
    worst = np.argsort(diff)[-5:][::-1]
    for idx in worst:
        print("Rated: {:3.1f}       Pred: {:5.3f}".format(score[idx], pred[idx][0]))
        print(test[idx])
