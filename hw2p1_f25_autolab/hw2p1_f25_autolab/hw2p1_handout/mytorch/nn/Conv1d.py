# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        N, Cin, Win = A.shape
        Cout = self.out_channels
        K = self.kernel_size
        Wout = Win - K + 1
        Z = np.zeros((N, Cout, Wout))
        for i in range(Wout):
            patch = self.A[:, :, i:i + K]
            Z[:, :, i] = np.tensordot(patch, self.W, ((1, 2), (1, 2)))
        Z += self.b[None, :, None]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        N, Cout, Wout = dLdZ.shape
        Cin = self.in_channels
        K = self.kernel_size
        Win = self.A.shape[2]

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        self.dLdW = np.zeros((Cout, Cin, K))
        for i in range(Wout):
            patch = self.A[:, :, i:i + K]
            self.dLdW += np.tensordot(dLdZ[:, :, i], patch, ((0), (0)))

        dLdA = np.zeros((N, Cin, Win))
        for i in range(Wout):
            temp = np.tensordot(dLdZ[:, :, i], self.W, ((1), (0)))
            dLdA[:, :, i:i + K] += temp

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')

        # Call Conv1d_stride1
        out = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(out)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        delta = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv1d_stride1.backward(delta)

        # Unpad the gradient
        dLdA = dLdA_padded[:, :, self.pad : -self.pad] if self.pad > 0 else dLdA_padded

        return dLdA