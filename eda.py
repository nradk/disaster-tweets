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
url_regex = re.compile("(http|https)://[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
# Compile regex to detect tokens that are entirely non-text
nontext_regex = re.compile("[^A-Za-z]+")
# Compile regex to detect @ mentions
mention_regex = re.compile("@\S+")
# Compile regex to detect various mis-encoded punctuation characters
misenc_regex = re.compile("&amp;");
# Compile regex to check if text is composed entirely of letters and digits
alphanum_regex = re.compile("[A-Za-z0-9]+");


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
    tweet_docs = train_df["text"].map(nlp)
    print("Generated in " + str(time.time() - start) + "s.")
    if not ALWAYS_COMPUTE:
        print("Saving tokenized data file...")
        start = time.time()
        tweet_docs.to_pickle(tokenized_pickle)
        print("Saved in " + str(time.time() - start) + "s.")

def clean(token_list):
    token_filters = [
            lambda t: url_regex.search(t.lemma_.lower()) is None,
            lambda t: nontext_regex.fullmatch(t.lemma_.lower()) is None,
            lambda t: mention_regex.match(t.lemma_.lower()) is None,
            ]
    # Apply filters
    token_list = filter(lambda t: all([f(t) for f in token_filters]),
            token_list)

    # Split possible joined words (like #HashTags)
    new_token_list = []
    is_oov = nlp.vocab.has_vector
    for token in token_list:
        if not is_oov(token.text) and not is_oov(token.lemma_):
            split = wordninja.split(token.lemma_.lower())
            print(token.text, ":", split)
            if all([nlp.vocab.has_vector(w) or alphanum_regex.fullmatch(w)
                    for w in split]):
                new_token_list.extend(split)
        else:
            new_token_list.append(token.lemma_.lower())
    return new_token_list

# Clean tokens
tweet_texts = [clean(tweet_doc) for tweet_doc in tweet_docs]

# Make a grand list of all tokens by flattening above list
all_words = [token for tweet in tweet_texts for token in tweet]

# Count words
word_counter = Counter(all_words)
sorted_words_with_counts = word_counter.most_common()
print(sorted_words_with_counts)
print("Length of word counter is", len(word_counter))
