#!/usr/bin/env python3

# Import libraries
import numpy as np
import pandas as pd
import spacy
import os.path
import time
from collections import Counter
import re
import matplotlib.pyplot as plt

DATA_DIR = 'data/'

train_file = DATA_DIR + 'train.csv'
test_file = DATA_DIR + 'test.csv'
sample_file = DATA_DIR + 'sample_submission.csv'
tokenized_pickle = DATA_DIR + 'tokenized.pkl'

# Load training and test CSV files
train_df = pd.read_csv(train_file, header=0)
test_df = pd.read_csv(test_file, header=0)
sample_df = pd.read_csv(sample_file, header=0)

# Load the SpaCy English language model trained on OntoNotes
nlp = spacy.load("en_core_web_sm")

# Load tokenized tweets if we have already saved them
tweet_docs = None
if os.path.isfile(tokenized_pickle):
    print("Saved tokenized data file found, loading...")
    start = time.time()
    tweet_docs = pd.read_pickle(tokenized_pickle)
    print("Loaded in " + str(time.time() - start) + "s.")
else:
    print("Saved tokenized data file not found, generating...")
    start = time.time()
    tweet_docs = train_df["text"].map(nlp)
    print("Generated in " + str(time.time() - start) + "s.")
    print("Saving tokenized data file...")
    start = time.time()
    tweet_docs.to_pickle(tokenized_pickle)
    print("Saved in " + str(time.time() - start) + "s.")

# Extract just the text from tokens
tweet_token_texts = [[t.lemma_.lower() for t in doc] for doc in tweet_docs]

# Make a grand list of all tokens by flattening above list
all_tokens = [token for tweet in tweet_token_texts for token in tweet]

# Count tokens
token_counter = Counter(all_tokens)
sorted_tokens_with_counts = token_counter.most_common()
print(sorted_tokens_with_counts)
print("Length of token counter is", len(token_counter))
