import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        N, Cin, Hin, Win = A.shape
        Cout = self.out_channels
        K = self.kernel_size
        Hout = Hin - K + 1
        Wout = Win - K + 1
        Z = np.zeros((N, Cout, Hout, Wout))
        for i in range(Hout):
            for j in range(Wout):
                patch = self.A[:, :, i:i + K, j:j + K]
                Z[:, :, i, j] = np.tensordot(patch, self.W, ((1, 2, 3), (1, 2, 3)))
        Z += self.b[None, :, None, None]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, Cout, Hout, Wout = dLdZ.shape
        Cin = self.in_channels
        K = self.kernel_size
        Hin, Win = self.A.shape[2:]

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        self.dLdW = np.zeros((Cout, Cin, K, K))
        for i in range(Hout):
            for j in range(Wout):
                patch = self.A[:, :, i:i + K, j:j + K]
                self.dLdW += np.tensordot(dLdZ[:, :, i, j], patch, ((0), (0)))

        dLdA = np.zeros((N, Cin, Hin, Win))
        for i in range(Hout):
            for j in range(Wout):
                temp = np.tensordot(dLdZ[:, :, i, j], self.W, ((1), (0)))
                dLdA[:, :, i:i + K, j:j + K] += temp

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() instance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

        # Call Conv2d_stride1
        out = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(out)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample2d backward
        delta = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA_padded = self.conv2d_stride1.backward(delta)

        # Unpad the gradient
        dLdA = dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad] if self.pad > 0 else dLdA_padded

        return dLdA