#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn

import preprocess

X, y = None, None

if preprocess.have_saved_vectors():
    X, y = preprocess.load_saved_vectors()
else:
    preprocess.load_data()
    preprocess.clean_data()
    X, y = preprocess.convert_to_numpy(save_to_disk=True)

# Convert the numpy arrays to torch vectors
X, y = torch.from_numpy(X), torch.from_numpy(y)

print(X.size())
print(y.size())
