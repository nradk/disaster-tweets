#!/usr/bin/env python3

import numpy as np
import torch
import sklearn
import sklearn.model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

import preprocess.glove_vectorize as glove_vectorize
import models
import load


# Load training and test data in Pandas dataframes
X_train_df, y_train_df = load.load_train_data()
X_test_df = load.load_test_data()


# GLOVE + CNN
# Preprocess training set and convert to torch tensor
X_train = glove_vectorize.preprocess_train(X_train_df)
X_train = torch.from_numpy(X_train)
y_train = y_train_df.to_numpy()
y_train = torch.from_numpy(y_train)
# Type cast to float tensor (our classifier doesn't seem to work with the
# default double tensor)
X_train = X_train.type(dtype=torch.FloatTensor)
y_train = y_train.type(dtype=torch.FloatTensor)
# Convert the target vector to one-hot encoding

sentence_length, vector_size = glove_vectorize.get_instance_dims()
cnn_model = models.CNNModel(vector_size=vector_size,
                            sentence_length=sentence_length)
clf = cnn_model.get_sklearn_compatible_estimator()

# clf.fit(X_train, y_train_df)
# exit()

print("Performing cross-validation")
scores = sklearn.model_selection.cross_val_score(clf, X_train, y_train_df, cv=5)
print("Cross-validation scores:")
print(scores)
print("Average cross-validation score:", sum(scores)/len(scores))
