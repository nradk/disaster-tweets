#!/usr/bin/env python3

import numpy as np
import torch
import sklearn
import sklearn.model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

import preprocess.glove_vectorize as glove_vectorize
import models
import load


# Load training and test data in Pandas dataframes
X_train_df, y_train_df = load.load_train_data()
X_test_df = load.load_test_data()
# X_train_df, y_train_df = X_train_df[:1000], y_train_df[:1000]


# Preprocess training set and convert to torch tensor
X_train = glove_vectorize.preprocess_train(X_train_df)
y_train = y_train_df.to_numpy()


def glove_cnn_run(X_train, y_train):
    global y_train_df
    # GLOVE + CNN
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    # Type cast to float tensor (our classifier doesn't seem to work with the
    # default double tensor)
    X_train = X_train.type(dtype=torch.FloatTensor)
    y_train = y_train.type(dtype=torch.FloatTensor)

    sentence_length, vector_size = glove_vectorize.get_instance_dims()
    cnn_model = models.CNNModel(vector_size=vector_size,
                                sentence_length=sentence_length)
    clf = cnn_model.get_sklearn_compatible_estimator()

    print("Performing cross-validation")
    scores = sklearn.model_selection.cross_val_score(
        clf, X_train, y_train_df, cv=5)
    print("Cross-validation scores:")
    print(scores)
    print("Average cross-validation score:", sum(scores)/len(scores))


def glove_lstm_run(X_train, y_train):
    global y_train_df
    # GLOVE + LSTM
    # Convert to a format suitable for RNNs
    X_train = glove_vectorize.get_as_word_vector_sequences(X_train)
    X_train = list(map(torch.from_numpy, X_train))
    y_train = torch.from_numpy(y_train)
    # Type cast to float tensor (our classifier doesn't seem to work with the
    # default double tensor)
    X_train = list(map(lambda x: x.type(dtype=torch.FloatTensor), X_train))
    y_train = y_train.type(dtype=torch.FloatTensor)

    X_train = torch.nn.utils.rnn.pad_sequence(X_train, padding_value=0,
                                              batch_first=True)

    _, vector_size = glove_vectorize.get_instance_dims()
    lstm_model = models.LSTMModel(vector_size=vector_size)
    clf = lstm_model.get_sklearn_compatible_estimator()

    print("Performing cross-validation")
    scores = sklearn.model_selection.cross_val_score(clf, X_train, y_train_df,
                                                     cv=3)
    print("Cross-validation scores:")
    print(scores)
    print("Average cross-validation score:", sum(scores)/len(scores))


def glove_knn_run(X_train, y_train):
    global y_train_df
    X_train = np.sum(X_train, axis=1)
    knn_classifier = KNeighborsClassifier()

    print("Performing cross-validation")
    scores = sklearn.model_selection.cross_val_score(knn_classifier, X_train,
                                                     y_train_df, cv=3)
    print("Cross-validation scores:")
    print(scores)
    print("Average cross-validation score:", sum(scores)/len(scores))


glove_knn_run(X_train, y_train)
