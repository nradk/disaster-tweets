#!/usr/bin/env python3

import numpy as np
import torch
import sklearn
import sklearn.model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import functools

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


def get_cnn_classifier():
    # GLOVE + CNN
    to_tensor = FunctionTransformer(torch.from_numpy)
    # Type cast to float tensor (our classifier doesn't seem to work with the
    # default double tensor)
    to_float = FunctionTransformer(lambda t: t.type(dtype=torch.FloatTensor))

    sentence_length, vector_size = glove_vectorize.get_instance_dims()
    cnn_model = models.CNNModel(vector_size=vector_size,
                                sentence_length=sentence_length)
    classifier = cnn_model.get_sklearn_compatible_estimator()
    return Pipeline([('to_tensor', to_tensor), ('to_float', to_float),
                     ('cnn', classifier)])


def get_lstm_classifier():
    global y_train_df
    # Convert to a format suitable for RNNs
    to_tensor = FunctionTransformer(torch.from_numpy)
    # Type cast to float tensor (our classifier doesn't seem to work with the
    # default double tensor)
    to_float = FunctionTransformer(lambda t: t.type(dtype=torch.FloatTensor))

    _, vector_size = glove_vectorize.get_instance_dims()
    lstm_model = models.LSTMModel(vector_size=vector_size)
    return Pipeline([('to_tensor', to_tensor), ('to_float', to_float),
                     ('lstm', lstm_model.get_sklearn_compatible_estimator())])


def get_knn_classifier():
    global y_train_df
    sum_vectors = FunctionTransformer(functools.partial(np.sum, axis=1))
    return Pipeline([('sum_vectors', sum_vectors),
                     ('knn', KNeighborsClassifier())])


ensemble = VotingClassifier(estimators=[('cnn', get_cnn_classifier()),
                                        ('lstm', get_lstm_classifier()),
                                        ('knn', get_knn_classifier())])

print("Performing cross-validation")
scores = sklearn.model_selection.cross_val_score(ensemble, X_train,
                                                 y_train_df, cv=3)
print("Cross-validation scores:")
print(scores)
print("Average cross-validation score:", sum(scores)/len(scores))
