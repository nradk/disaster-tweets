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

DATA_DIR = 'data/'
ALWAYS_COMPUTE = True

train_file = DATA_DIR + 'train.csv'
test_file = DATA_DIR + 'test.csv'
sample_file = DATA_DIR + 'sample_submission.csv'
tokenized_pickle = DATA_DIR + 'tokenized.pkl'

# Load training and test CSV files
train_df = pd.read_csv(train_file, header=0)
test_df = pd.read_csv(test_file, header=0)
sample_df = pd.read_csv(sample_file, header=0)

# Load the SpaCy English language model trained on OntoNotes with GloVe vectors
# trained on Common Crawl.
nlp = spacy.load("en_core_web_lg")

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
# Compile regex to check if text is a hashtag


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


# Load tokenized tweets if we have already saved them
tweet_docs = None
if os.path.isfile(tokenized_pickle) and not ALWAYS_COMPUTE:
    print("Saved tokenized data file found, loading...")
    start = time.time()
    tweet_docs = pd.read_pickle(tokenized_pickle)
    print("Loaded in " + str(time.time() - start) + "s.")
else:
    print("Saved tokenized data file not found, generating...")
    start = time.time()
    # Perform pre-tokenization cleaning
    train_df["text"] = train_df["text"].map(clean_pre_tokenize)
    # Perform tokenization
    tweet_docs = train_df["text"].map(nlp)
    print("Generated in " + str(time.time() - start) + "s.")
    if not ALWAYS_COMPUTE:
        print("Saving tokenized data file...")
        start = time.time()
        tweet_docs.to_pickle(tokenized_pickle)
        print("Saved in " + str(time.time() - start) + "s.")

# Clean tokens
train_df["text"] = [list(clean_post_tokenize(tweet_doc)) for tweet_doc in
        tweet_docs]
print(train_df.head())
