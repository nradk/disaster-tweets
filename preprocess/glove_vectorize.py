#!/usr/bin/env python3

# Import libraries
import numpy as np
import pandas as pd
import spacy
import os.path
import time
from collections import Counter
import re
import wordninja
import functools
import sklearn

import load

DATA_DIR = 'data/'

train_vectors_file = DATA_DIR + 'train_vectors.npz'
test_vectors_file = DATA_DIR + 'test_vectors.npz'
train_hash_file = DATA_DIR + 'train_vectors_hash.txt'
test_hash_file = DATA_DIR + 'test_vectors_hash.txt'

# Compile url regex (a basic one)
url_regex = re.compile("(http|https)://[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(/\S*)?")
# Compile regex to detect tokens that are entirely non-text
nontext_regex = re.compile("[^A-Za-z]+")
# Compile regex to detect @ mentions
mention_regex = re.compile("@\S+")
# Compile regex to detect various mis-encoded punctuation characters
misenc_regex = re.compile("&amp;")
# Compile regex to check if text is composed entirely of letters and digits
alphanum_regex = re.compile("[A-Za-z0-9]+")


def load_nlp_model():
    # Load the SpaCy English language model trained on OntoNotes with GloVe
    # vectors trained on Common Crawl.
    return spacy.load("en_core_web_lg")


def write_hash_to_file(filename, hash_value):
    with open(filename, 'w') as f:
        f.write("%d" % hash_value)


def read_hash_from_file(filename):
    hash_value = -1
    with open(filename, 'r') as f:
        hash_value = int(f.read())
    return hash_value


def have_saved_vectors(df_hash, test=False):
    hash_file = test_hash_file if test else train_hash_file
    vectors_file = test_vectors_file if test else train_vectors_file
    return (os.path.isfile(vectors_file) and os.path.isfile(hash_file) and
            df_hash == read_hash_from_file(hash_file))


def load_saved_vectors(filename, *keys):
    print("Loading saved vectors")
    start = time.time()
    npzfile = np.load(filename)
    print("Loaded saved vectors in", time.time() - start, "seconds")
    return tuple([npzfile[key] for key in keys])


def save_vectors(filename, **kwargs):
    print("Saving computed vectors to disk")
    start = time.time()
    np.savez_compressed(filename, **kwargs)
    print("Saved computed vectors to disk in", time.time() - start, "seconds")


# Data cleaning prior to tokenization
def clean_pre_tokenize(text, nlp):
    text = url_regex.sub("", text)
    text = misenc_regex.sub("", text)
    text = mention_regex.sub("", text)
    text = re.sub("#", "", text)
    words = text.split(" ")
    split_words = []
    for word in words:
        if not nlp.vocab.has_vector(word):
            split_words.extend(wordninja.split(word))
        else:
            split_words.append(word)
    return " ".join(split_words)


# Data cleaning after tokenization (takes token list and returns word list)
def clean_post_tokenize(token_list):
    # Filter the list to remove non-text tokens and return result
    return filter(lambda t: not t.is_oov and not t.is_stop, token_list)


def clean_data_(X_df, nlp):
    print("Performing pre-tokenization cleaning...")
    start = time.time()
    # Perform pre-tokenization cleaning
    X_df["text"] = X_df["text"].map(lambda text: clean_pre_tokenize(text, nlp))
    print("Cleaned in", (time.time() - start), "seconds. Now tokenizing...")
    start = time.time()
    # Perform tokenization
    tweet_docs = X_df["text"].map(nlp)
    print("Tokenized in " + str(time.time() - start) + "s.")

    # Clean tokens
    print("Performing post-tokenization cleaning...")
    start = time.time()
    X_df["text"] = [list(clean_post_tokenize(tweet_doc)) for tweet_doc in
                    tweet_docs]
    print("Cleaned in in " + str(time.time() - start) + "s.")


def convert_to_numpy(df):
    # Convert tokens to their vector representations and save them as numpy
    # arrays
    df["text"] = df["text"].map(
        lambda l: np.array(list(map(lambda t: t.vector, l))))
    # Get the size of a single word vector
    vector_size = df["text"][0].shape[1]
    # Find out the maximum length of the sentences (in words/tokens)
    # max_length = max(df["text"].map(lambda arr: len(arr)))
    # TODO sort this out (using a fixed max length now)
    max_length = 34

    # Resize the token-vector array so that all are of the same length (with
    # zero padding)
    df["text"] = df["text"].map(lambda arr:
                                np.resize(arr, (max_length, vector_size)))

    # Create and fill a numpy array with the entire input matrix row-by-row
    # because pandas to_numpy() seems to produce weird results
    X = np.ndarray((len(df), max_length, vector_size))
    for i in range(X.shape[0]):
        X[i, :] = df["text"][i]
    return X


def get_df_hash(df):
    return pd.util.hash_pandas_object(df).sum()


def preprocess_train(X_train_df, use_saved_vectors=True):
    X, y = None, None
    df_hash = get_df_hash(X_train_df)
    if use_saved_vectors and have_saved_vectors(df_hash, test=False):
        X, = load_saved_vectors(train_vectors_file, "X")
    else:
        nlp = load_nlp_model()
        clean_data_(X_train_df, nlp)
        X = convert_to_numpy(X_train_df)
        save_vectors(train_vectors_file, X=X)
        write_hash_to_file(train_hash_file, df_hash)
    return X


def preprocess_test(X_test_df, use_saved_vectors=True):
    X = None
    df_hash = get_df_hash(X_test_df)
    if use_saved_vectors and have_saved_vectors(df_hash, test=True):
        X, = load_saved_vectors(test_vectors_file, "X")
    else:
        nlp = load_nlp_model()
        clean_data_(X_test_df, nlp, test=True)
        X = convert_to_numpy(X_test_df)
        save_vectors(test_vectors_file, X=X)
        write_hash_to_file(test_hash_file, df_hash)
    return X


def get_transformer_train():
    return sklearn.preprocessing.FunctionTransformer(preprocess_train)


def get_transformer_test():
    return sklearn.preprocessing.FunctionTransformer(preprocess_test)


def get_instance_dims():
    return (34, 300)
