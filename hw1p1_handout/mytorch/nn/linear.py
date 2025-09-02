
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        W shape: (C_out, C_in) = (out_features, in_features)
        b shape: (C_out, 1) = (out_features, 1)
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features))  # Shape: (C_out, C_in)
        self.b = np.zeros((out_features, 1))            # Shape: (C_out, 1)

    def forward(self, A):
        """
        Forward pass: Z = A · W^T + ι_N · b^T
        :param A: Input with shape (N, C_in)
        :return: Output Z with shape (N, C_out)
        """
        self.A = A  # Store for backward pass
        self.N = A.shape[0]  # Store batch size
        
        # Equation (1) from writeup: Z = A · W^T + ι_N · b^T
        # A: (N, C_in), W^T: (C_in, C_out) -> A @ W^T: (N, C_out)
        # ι_N: (N, 1), b^T: (1, C_out) -> ι_N @ b^T: (N, C_out)
        Z = A @ self.W.T + np.ones((self.N, 1)) @ self.b.T
        
        return Z

    def backward(self, dLdZ):
        """
        Backward pass to compute gradients
        :param dLdZ: Gradient of loss wrt output Z with shape (N, C_out)
        :return: Gradient of loss wrt input A with shape (N, C_in)
        """
        # Equation (5): ∂L/∂A = (∂L/∂Z) · W
        dLdA = dLdZ @ self.W
        
        # Equation (6): ∂L/∂W = (∂L/∂Z)^T · A
        self.dLdW = dLdZ.T @ self.A
        
        # Equation (7): ∂L/∂b = (∂L/∂Z)^T · ι_N
        self.dLdb = dLdZ.T @ np.ones((self.N, 1))
        
        if self.debug:
            self.dLdA = dLdA
            
        return dLdA