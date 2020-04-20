#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

N = X.size()[0]
L = X.size()[1]
C = X.size()[2]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # First convolution layer.  C input channels, 4 output channels and
        # convolution kernel size 2.
        self.conv1 = nn.Conv1d(C, 4, 3, padding=1)
        # Second convolution layer.  4 input channels from the previous layer,
        # 12 output channels and convolution kernel size 3.
        self.conv2 = nn.Conv1d(4, 8, 5, padding=2)
        # Fuly connected layers
        self.fc1 = nn.Linear(8 * L, 40)
        self.fc2 = nn.Linear(40, 2)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
net.float()
net.zero_grad()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

EPOCHS = 5
for epoch in range(EPOCHS):
    for i in range(N):
        optimizer.zero_grad()
        target = torch.unsqueeze(y[i], 0)
        output = net(torch.unsqueeze(torch.transpose(X[i,:], 0, 1), 0).float())
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch {} done".format(epoch))
