#!/usr/bin/env python3

import numpy as np
import torch

import preprocess
import models

Xtrain, ytrain = None, None
Xtest = None

Xtrain, ytrain = preprocess.load_and_preprocess_train(use_saved_vectors=True)

# Convert the numpy arrays to torch vectors
X, y = torch.from_numpy(Xtrain[:6000, :, :]), torch.from_numpy(ytrain[:6000])
Xv, yv = torch.from_numpy(Xtrain[6000:, :, :]), torch.from_numpy(ytrain[6000:])


N = X.size()[0]
L = X.size()[1]
C = X.size()[2]

cnn_model = models.CNNModel(channels=C, size=L)
cnn_model.learn(X, y)

predicted = cnn_model.infer(Xv)
total = yv.size()[0]
correct = (predicted == yv).sum().item()
print("Correct =", correct)
print("Total =", total)
print("Accuracy =", (correct/total))
