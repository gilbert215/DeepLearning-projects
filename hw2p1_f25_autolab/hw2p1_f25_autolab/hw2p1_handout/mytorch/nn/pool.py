import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        
        # Calculate output dimensions
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        
        # Initialize output
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        
        # Store indices for backward pass (for gradient routing)
        self.max_indices = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=int)
        
        # Perform max pooling
        for h in range(output_height):
            for w in range(output_width):
                # Extract the kernel-sized patch
                patch = A[:, :, h:h+self.kernel, w:w+self.kernel]
                
                # Find max values along the spatial dimensions
                # Reshape patch to (batch_size, in_channels, kernel*kernel)
                patch_reshaped = patch.reshape(batch_size, in_channels, -1)
                max_vals = np.max(patch_reshaped, axis=2)
                Z[:, :, h, w] = max_vals
                
                # Find indices of max values for backward pass
                for b in range(batch_size):
                    for c in range(in_channels):
                        patch_2d = patch[b, c, :, :]
                        max_idx = np.unravel_index(np.argmax(patch_2d), patch_2d.shape)
                        self.max_indices[b, c, h, w, 0] = h + max_idx[0]
                        self.max_indices[b, c, h, w, 1] = w + max_idx[1]
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, input_height, input_width = self.A.shape
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        
        output_height, output_width = dLdZ.shape[2], dLdZ.shape[3]
        
        # Route gradients back to the positions that had maximum values
        for h in range(output_height):
            for w in range(output_width):
                for b in range(batch_size):
                    for c in range(in_channels):
                        max_h = self.max_indices[b, c, h, w, 0]
                        max_w = self.max_indices[b, c, h, w, 1]
                        dLdA[b, c, max_h, max_w] += dLdZ[b, c, h, w]
        
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        
        # Calculate output dimensions
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1
        
        # Initialize output
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        
        # Perform mean pooling
        for h in range(output_height):
            for w in range(output_width):
                # Extract the kernel-sized patch
                patch = A[:, :, h:h+self.kernel, w:w+self.kernel]
                
                # Calculate mean over the spatial dimensions
                Z[:, :, h, w] = np.mean(patch, axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, input_height, input_width = self.A.shape
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        
        output_height, output_width = dLdZ.shape[2], dLdZ.shape[3]
        
        # Distribute gradients equally to all positions in each kernel patch
        for h in range(output_height):
            for w in range(output_width):
                # Each element in the kernel patch gets an equal share of the gradient
                gradient_per_element = dLdZ[:, :, h, w] / (self.kernel * self.kernel)
                
                # Add this gradient to all positions in the corresponding patch
                dLdA[:, :, h:h+self.kernel, w:w+self.kernel] += gradient_per_element[:, :, np.newaxis, np.newaxis]
        
        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # First apply max pooling with stride 1
        pooled = self.maxpool2d_stride1.forward(A)
        
        # Then downsample by the stride factor
        Z = self.downsample2d.forward(pooled)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Backward through downsampling first
        dLdPooled = self.downsample2d.backward(dLdZ)
        
        # Then backward through max pooling
        dLdA = self.maxpool2d_stride1.backward(dLdPooled)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # First apply mean pooling with stride 1
        pooled = self.meanpool2d_stride1.forward(A)
        
        # Then downsample by the stride factor
        Z = self.downsample2d.forward(pooled)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Backward through downsampling first
        dLdPooled = self.downsample2d.backward(dLdZ)
        
        # Then backward through mean pooling
        dLdA = self.meanpool2d_stride1.backward(dLdPooled)
        
        return dLdA