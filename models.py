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
    methods `learn()` and `infer()`.
    """

    def learn(self, input_data, labels):
        """
        Learn a representation from the given input data and labels.

        :param input_data: Input data to learn from, a pytorch tensor.
        :param labels: Lablels for the input, a pytorch tensor.
        """
        pass

    def infer(self, input_data):
        """
        Use the model to perform inference on input.

        :param input_data: The input to perform inference on. Pytorch tensor.
        :returns: A pytorch tensor containing inference outputs.
        """
        pass

    def get_sklearn_compatible_estimator(self):
        """
        Get a scikit-learn compatible estimator that implements the
        scikit-learn base estimator API. Can be either an actual scikit-learn
        class or a compatible implementation like from skorch.
        """
        pass


class CNNModel(Model):
    """
    Implements a Convolutional Neural Network

    Implements a small convolutional neural network with two convolutional
    layers and two fully connected layers. See inner class `Net` for
    implementation details.
    """

    class Net(nn.Module):

        def __init__(self, channels, size):
            super(CNNModel.Net, self).__init__()
            # First convolution layer.  C input channels, 4 output channels and
            # convolution kernel size 3.
            self.conv1 = nn.Conv1d(channels, 4, 3, padding=1)
            # Second convolution layer. 4 input channels from the previous
            # layer, 8 output channels and convolution kernel size 5.
            self.conv2 = nn.Conv1d(4, 8, 5, padding=2)
            # Fuly connected layers
            self.fc1 = nn.Linear(8 * size, 40)
            self.fc2 = nn.Linear(40, 2)

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

    def __init__(self, channels, size):
        self.channels = channels
        self.size = size
        self.net = self.Net(channels, size)
        self.net.float()
        self.criterion = nn.CrossEntropyLoss
        self.loss_function = self.criterion()
        self.optimizer_class = optim.Adam
        self.optimizer = self.optimizer_class(self.net.parameters())
        self.EPOCHS = 7

    def learn(self, input_data, labels):
        self.net.zero_grad()
        print("Starting CNN training.")
        X, y = input_data, labels
        N = input_data.size()[0]
        for epoch in range(self.EPOCHS):
            for i in range(N):
                self.optimizer.zero_grad()
                target = torch.unsqueeze(y[i], 0)
                output = self.net(torch.unsqueeze(X[i,:,:], 0))
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
            print("Epoch {} done".format(epoch))

    def infer(self, input_data):
        _, predicted = torch.max(self.net(input_data), 1)
        return predicted

    def get_sklearn_compatible_estimator(self):
        net_with_params = functools.partial(self.Net, channels=self.channels,
                                            size=self.size)
        return skorch.NeuralNetClassifier(net_with_params,
                                          max_epochs=self.EPOCHS,
                                          criterion=self.criterion,
                                          optimizer=self.optimizer_class)
