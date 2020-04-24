#!/usr/bin/env python3

import numpy as np
import torch
import sklearn
import sklearn.model_selection

import preprocess
import models

Xtrain, ytrain = preprocess.load_and_preprocess_train(use_saved_vectors=True)
Xtrain, ytrain = torch.from_numpy(Xtrain), torch.from_numpy(ytrain)

# Change types of all tensors to float (leaving double produces an obscure
# error)
Xtrain = Xtrain.type(dtype=torch.FloatTensor)
y = ytrain.type(dtype=torch.FloatTensor)

# Transpose the last two dimensions of Xtrain so that the elements of word
# vectors become the channels.
Xtrain.transpose_(1, 2)

N = Xtrain.size()[0]
C = Xtrain.size()[1]
L = Xtrain.size()[2]

cnn_model = models.CNNModel(channels=C, size=L)

print("Performing cross-validation")
clf = cnn_model.get_sklearn_compatible_estimator()
scores = sklearn.model_selection.cross_val_score(clf, Xtrain, ytrain, cv=5)
print("Cross-validation scores:")
print(scores)
print("Average cross-validation score:", sum(scores)/len(scores))
