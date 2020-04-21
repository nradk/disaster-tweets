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

DATA_DIR = 'data/'

train_file = DATA_DIR + 'train.csv'
test_file = DATA_DIR + 'test.csv'
sample_file = DATA_DIR + 'sample_submission.csv'
train_vectors_file = DATA_DIR + 'train_vectors.npz'
test_vectors_file = DATA_DIR + 'test_vectors.npz'

# Compile url regex (a basic one)
url_regex = re.compile("(http|https)://[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(/\S*)?")
# Compile regex to detect tokens that are entirely non-text
nontext_regex = re.compile("[^A-Za-z]+")
# Compile regex to detect @ mentions
mention_regex = re.compile("@\S+")
# Compile regex to detect various mis-encoded punctuation characters
misenc_regex = re.compile("&amp;");
# Compile regex to check if text is composed entirely of letters and digits
alphanum_regex = re.compile("[A-Za-z0-9]+");

train_df = None
test_df = None
nlp = None


def load_data(test=False):
    if test:
        global test_df
        test_df = pd.read_csv(test_file, header=0)
    else:
        global train_df
        train_df = pd.read_csv(train_file, header=0)


def load_nlp_model():
    global nlp
    # Load the SpaCy English language model trained on OntoNotes with GloVe
    # vectors trained on Common Crawl.
    nlp = spacy.load("en_core_web_lg")


def have_saved_vectors(test=False):
    return os.path.isfile(test_vectors_file if test else train_vectors_file)


def load_saved_vectors(test=False):
    print("Loading saved vectors")
    start = time.time()
    npzfile = np.load(test_vectors_file if test else train_vectors_file)
    print("Loaded saved vectors in", time.time() - start, "seconds")
    return (npzfile['X'], npzfile['y']) if not test else npzfile['X']


# Data cleaning prior to tokenization
def clean_pre_tokenize(text):
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


def clean_data(test=False):
    global train_df, test_df, nlp
    df = test_df if test else train_df
    print("Performing pre-tokenization cleaning...")
    start = time.time()
    # Perform pre-tokenization cleaning
    df["text"] = df["text"].map(clean_pre_tokenize)
    print("Cleaned in", (time.time() - start), "seconds. Now tokenizing...")
    start = time.time()
    # Perform tokenization
    tweet_docs = df["text"].map(nlp)
    print("Tokenized in " + str(time.time() - start) + "s.")

    # Clean tokens
    print("Performing post-tokenization cleaning...")
    start = time.time()
    df["text"] = [list(clean_post_tokenize(tweet_doc)) for tweet_doc in
            tweet_docs]
    print("Cleaned in in " + str(time.time() - start) + "s.")


def convert_to_numpy(test=False, save_to_disk=False):
    global train_df, test_df
    df = test_df if test else train_df
    # Convert tokens to their vector representations and save them as numpy
    # arrays
    df["text"] = df["text"].map(
            lambda l: np.array(list(map(lambda t: t.vector, l))))
    # Get the size of a single word vector
    vector_size = df["text"][0].shape[1]
    # Find out the maximum length of the sentences (in words/tokens)
    max_length = max(df["text"].map(lambda arr: len(arr)))

    # Resize the token-vector array so that all are of the same length (with
    # zero padding)
    df["text"] = df["text"].map(lambda arr: np.resize(arr,
        (max_length, vector_size)))

    # Create and fill a numpy array with the entire input matrix row-by-row
    # because pandas to_numpy() seems to produce weird results
    X = np.ndarray((len(df), max_length, vector_size))
    for i in range(X.shape[0]):
        X[i,:] = df["text"][i]
    # Get the targets as a numpy array

    if not test:
        y = df["target"].to_numpy()

    if save_to_disk:
        print("Saving computed vectors to disk")
        start = time.time()
        if test:
            np.savez_compressed(test_vectors_file, X=X)
        else:
            np.savez_compressed(train_vectors_file, X=X, y=y)
        print("Saved computed vectors to disk in", time.time() - start,
            "seconds")
    return X,y if not test else X


def load_and_preprocess_train(use_saved_vectors=True):
    X, y = None, None
    if use_saved_vectors and have_saved_vectors():
        X, y = load_saved_vectors()
    else:
        load_data()
        load_nlp_model()
        clean_data()
        X, y = convert_to_numpy(save_to_disk=True)
    return X, y


def load_and_preprocess_test(use_saved_vectors=True):
    X = None
    if use_saved_vectors and have_saved_vectors(test=True):
        X = load_saved_vectors(test=True)
    else:
        load_data(test=True)
        load_nlp_model()
        clean_data(test=True)
        X = convert_to_numpy(test=True,  save_to_disk=True)
    return X


if __name__ == "__main__":
    print(load_and_preprocess_train(True))
