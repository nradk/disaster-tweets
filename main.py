#!/usr/bin/env python3

import numpy as np
import torch
import sklearn
import sklearn.model_selection

import preprocess.glove_vectorize as glove_vectorize
import models
import load

Xtrain, ytrain = glove_vectorize.preprocess_train(*load.load_train_data(),
                                                  use_saved_vectors=True)
Xtrain, ytrain = torch.from_numpy(Xtrain), torch.from_numpy(ytrain)

# Change types of all tensors to float (leaving double produces an obscure
# error)
Xtrain = Xtrain.type(dtype=torch.FloatTensor)
y = ytrain.type(dtype=torch.FloatTensor)

cnn_model = models.CNNModel(vector_size=Xtrain.size()[2],
                            sentence_length=Xtrain.size()[1])

print("Performing cross-validation")
clf = cnn_model.get_sklearn_compatible_estimator()
scores = sklearn.model_selection.cross_val_score(clf, Xtrain, ytrain, cv=5)
print("Cross-validation scores:")
print(scores)
print("Average cross-validation score:", sum(scores)/len(scores))
