import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        self.net = self.Net(channels, size)
        self.net.float()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.EPOCHS = 1

    def learn(self, input_data, labels):
        self.net.zero_grad()
        print("Starting CNN training.")
        X, y = input_data, labels
        N = input_data.size()[0]
        for epoch in range(self.EPOCHS):
            for i in range(N):
                self.optimizer.zero_grad()
                target = torch.unsqueeze(y[i], 0)
                transposed_input = torch.transpose(X[i, :], 0, 1)
                output = self.net(torch.unsqueeze(transposed_input, 0).float())
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
            print("Epoch {} done".format(epoch))

    def infer(self, input_data):
        X = input_data
        _, predicted = torch.max(self.net(torch.transpose(X, 1, 2).float()), 1)
        return predicted
