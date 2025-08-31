import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros((in_features, out_features))  # TODO
        self.b = np.zeros((1, out_features))  # TODO

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A
        self.N = A.shape[0]

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        # self.ones = np.ones((self.N, 1))

        Z = A @ self.W + self.b
        return Z 
    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdA = dLdZ @ self.W.T
        self.dLdW = self.A.T @ dLdZ
        self.dLdb = np.sum(dLdZ, axis=0, keepdims=True)

        if self.debug:
            self.dLdA = dLdA
        return dLdA
