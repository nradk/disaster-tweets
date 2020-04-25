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

X_train_df, y_train_df = load.load_train_data()
y = torch.from_numpy(y_train_df.to_numpy()).type(dtype=torch.FloatTensor)
X_test_df = load.load_test_data()

glove_vectorizer = glove_vectorize.get_transformer_train()
torch_converter = FunctionTransformer(torch.from_numpy)
type_caster = FunctionTransformer(lambda t: t.type(dtype=torch.FloatTensor))

sentence_length, vector_size = glove_vectorize.get_instance_dims()
cnn_model = models.CNNModel(vector_size=vector_size,
                            sentence_length=sentence_length)
clf = cnn_model.get_sklearn_compatible_estimator()

glove_CNN_pipeline = Pipeline([
    ('glove_vectorizer', glove_vectorizer),
    ('torch_converter', torch_converter),
    ('type_caster', type_caster),
    ('cnn_classifier', clf),
])

glove_CNN_pipeline.fit(X_train_df, y_train_df)
print(glove_CNN_pipeline.predict(X_test_df))

# TODO K-fold cross validation is does not work with our pipeline, and even if
# it did work it would be very inefficient due to the word vectors having to be
# recalculated. Find a way to fix this.
# print("Performing cross-validation")
# scores = sklearn.model_selection.cross_val_score(glove_CNN_pipeline,
                                                 # X_train_df, y_train_df, cv=5)
# print("Cross-validation scores:")
# print(scores)
# print("Average cross-validation score:", sum(scores)/len(scores))
