import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import skorch
import functools


class Model:
    """
    Base class for machine learning models

    Implementations of models should sublcass this class and implement the
    `get_sklearn_compatible_estimator()` method.
    """

    def get_sklearn_compatible_estimator(self):
        """
        Get a scikit-learn compatible estimator that implements the
        scikit-learn base estimator API. Can be either an actual scikit-learn
        class or a compatible implementation like from skorch.
        """
        pass


class CNNModel(nn.Module, Model):
    """
    Implements a Convolutional Neural Network

    Implements a small convolutional neural network with two convolutional
    layers and two fully connected layers. See constructor and method
    `forward()` for implementation details.
    """

    def __init__(self, vector_size, sentence_length):
        super(CNNModel, self).__init__()
        self.channels = vector_size
        self.size = sentence_length
        # First convolution layer.  C input channels, 4 output channels and
        # convolution kernel size 3.
        self.conv1 = nn.Conv1d(self.channels, 4, 3, padding=1)
        # Second convolution layer. 4 input channels from the previous
        # layer, 8 output channels and convolution kernel size 5.
        self.conv2 = nn.Conv1d(4, 8, 5, padding=2)
        # Fuly connected layers
        self.fc1 = nn.Linear(8 * self.size, 40)
        self.fc2 = nn.Linear(40, 2)
        self.float()
        self.zero_grad()

    def forward(self, x):
        # print(x)
        # Transpose the last two dimensions of Xtrain so that the elements of
        # word vectors become the channels.
        dims = len(x.size())
        x = x.transpose(dims-2, dims-1)
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

    def get_sklearn_compatible_estimator(self):
        EPOCHS = 3
        criterion = nn.CrossEntropyLoss
        optimizer = optim.Adam
        net_with_params = functools.partial(CNNModel,
                                            vector_size=self.channels,
                                            sentence_length=self.size)
        return skorch.NeuralNetClassifier(net_with_params, max_epochs=EPOCHS,
                                          criterion=criterion,
                                          optimizer=optimizer)
