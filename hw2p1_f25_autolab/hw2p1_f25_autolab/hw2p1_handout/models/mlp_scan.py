from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # CNN equivalent of [Flatten(), Linear(8*24, 8), ReLU(), Linear(8, 16), ReLU(), Linear(16, 4)]
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)

        self.layers = [
            self.conv1, ReLU(),
            self.conv2, ReLU(),
            self.conv3
        ]

    def init_weights(self, weights):
        w1, w2, w3 = weights[0], weights[1], weights[2]

        # Layer 1: (192, 8) → (8, 8, 24) → (8, 24, 8)
        W1 = w1.T.reshape(8, 8, 24).transpose(0, 2, 1)
        self.conv1.conv1d_stride1.W = W1

        # Layer 2: (8, 16) → (16, 1, 8) → (16, 8, 1)
        W2 = w2.T.reshape(16, 1, 8).transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = W2

        # Layer 3: (16, 4) → (4, 1, 16) → (4, 16, 1)
        W3 = w3.T.reshape(4, 1, 16).transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = W3

    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Based on parameter sharing (Figure 27 in writeup)
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=2, out_channels=2, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        self.layers = [
            self.conv1, ReLU(),
            self.conv2, ReLU(),
            self.conv3
        ]

    def __call__(self, A):
        return self.forward(A)

    def init_weights(self, weights):
        w1, w2, w3 = weights[0], weights[1], weights[2]

        # Layer 1 (only 2 unique sets)
        W1 = w1.T.reshape(-1, 8, 24).transpose(0, 2, 1)
        W1 = W1[:2]  # keep only shared
        self.conv1.conv1d_stride1.W = W1

        # Layer 2 (only 2 unique sets)
        W2 = w2.T.reshape(-1, 1, 2).transpose(0, 2, 1)
        W2 = W2[:2]
        self.conv2.conv1d_stride1.W = W2

        # Layer 3 (only 1 unique set)
        W3 = w3.T.reshape(-1, 1, 2).transpose(0, 2, 1)
        W3 = W3[:1]
        self.conv3.conv1d_stride1.W = W3


    def forward(self, A):
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA

