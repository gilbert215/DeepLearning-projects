# import numpy as np
# import scipy


# ### No need to modify Identity class
# class Identity:
#     """
#     Identity activation function.
#     """

#     def forward(self, Z):
#         """
#         :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
#         :return: Output returns the computed output A (N samples, C features).
#         """
#         self.A = Z
#         return self.A

#     def backward(self, dLdA):
#         """
#         :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
#         :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
#         """
#         dAdZ = np.ones(self.A.shape, dtype="f")
#         dLdZ = dLdA * dAdZ
#         return dLdZ


# class Sigmoid:
#     """
#     Sigmoid activation function.

#     TODO:
#     On same lines as above, create your own mytorch.nn.Sigmoid!
#     Define 'forward' function.
#     Define 'backward' function.
#     Read the writeup (Hint: Sigmoid Section) for further details on Sigmoid forward and backward expressions.
#     """


# class Tanh:
#     """
#     Tanh activation function.

#     TODO:
#     On same lines as above, create your own mytorch.nn.Tanh!
#     Define 'forward' function.
#     Define 'backward' function.
#     Read the writeup (Hint: Tanh Section) for further details on Tanh forward and backward expressions.
#     """


# class ReLU:
#     """
#     ReLU (Rectified Linear Unit) activation function.

#     TODO:
#     On same lines as above, create your own mytorch.nn.ReLU!
#     Define 'forward' function.
#     Define 'backward' function.
#     Read the writeup (Hint: ReLU Section) for further details on ReLU forward and backward expressions.
#     """


# class GELU:
#     """
#     GELU (Gaussian Error Linear Unit) activation function.

#     TODO:
#     On same lines as above, create your own mytorch.nn.GELU!
#     Define 'forward' function.
#     Define 'backward' function.
#     Read the writeup (Hint: GELU Section) for further details on GELU forward and backward expressions.
#     Note: Feel free to save any variables from gelu.forward that you might need for gelu.backward.
#     """

# class Swish:
#     """
#     Swish activation function.

#     TODO:
#     On same lines as above, create your own Swish which is a torch.nn.SiLU with a learnable parameter (beta)!
#     Define 'forward' function.
#     Define 'backward' function.
#     Read the writeup (Hint: Swish Section) for further details on Swish forward and backward expressions.
#     """


# class Softmax:
#     """
#     Softmax activation function.

#     ToDO:
#     On same lines as above, create your own mytorch.nn.Softmax!
#     Complete the 'forward' function.
#     Complete the 'backward' function.
#     Read the writeup (Hint: Softmax Section) for further details on Softmax forward and backward expressions.
#     Hint: You read more about `axis` and `keep_dims` attributes, helpful for future homeworks too.
#     """

#     def forward(self, Z):
#         """
#         Remember that Softmax does not act element-wise.
#         It will use an entire row of Z to compute an output element.
#         Note: How can we handle large overflow values? Hint: Check numerical stability.
#         """
#         self.A = None  # TODO
#         raise NotImplementedError  # TODO - What should be the return value?

#     def backward(self, dLdA):
#         # Calculate the batch size and number of features
#         N = None  # TODO
#         C = None  # TODO

#         # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
#         dLdZ = None  # TODO

#         # Fill dLdZ one data point (row) at a time.
#         for i in range(N):
#             # Initialize the Jacobian with all zeros.
#             # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
#             J = None  # TODO

#             # Fill the Jacobian matrix, please read the writeup for the conditions.
#             for m in range(C):
#                 for n in range(C):
#                     J[m, n] = None  # TODO

#             # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
#             # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
#             dLdZ[i, :] = None  # TODO

#         raise NotImplementedError  # TODO - What should be the return value?


import numpy as np
import scipy
from scipy.special import erf


### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)  # sigmoid’(Z) = A - A^2
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    Tanh activation function.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.Z > 0, 1, 0)
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        Z = self.Z
        dAdZ = 0.5 * (1 + erf(Z / np.sqrt(2))) + (Z / np.sqrt(2 * np.pi)) * np.exp(-Z**2 / 2)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Swish:
    """
    Swish activation function with learnable parameter beta.
    """

    def __init__(self, beta=1.0):
        self.beta = beta

    def forward(self, Z):
        self.Z = Z
        self.sigmoid_betaZ = 1 / (1 + np.exp(-self.beta * Z))
        self.A = Z * self.sigmoid_betaZ
        return self.A

    def backward(self, dLdA):
        Z = self.Z
        σ = self.sigmoid_betaZ
        dAdZ = σ + self.beta * Z * σ * (1 - σ)
        dLdZ = dLdA * dAdZ
        # dA/dbeta = Z^2 * σ(βZ) * (1 - σ(βZ))
        dAdBeta = Z * Z * σ * (1 - σ)
        self.dLdbeta = np.sum(dLdA * dAdBeta)
        return dLdZ

    def grad_beta(self, dLdA):
        # ∂A/∂β = Z^2 * σ(βZ) * (1 - σ(βZ))
        Z = self.Z
        σ = self.sigmoid_betaZ
        dAdBeta = Z * Z * σ * (1 - σ)
        dLdBeta = np.sum(dLdA * dAdBeta)
        return dLdBeta


class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)  # numerical stability
        expZ = np.exp(Z_stable)
        self.A = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        N, C = self.A.shape
        dLdZ = np.zeros_like(dLdA)

        for i in range(N):
            Ai = self.A[i].reshape(-1, 1)  # (C, 1)
            J = np.diagflat(Ai) - Ai @ Ai.T  # Jacobian (C, C)
            dLdZ[i] = dLdA[i] @ J  # (1, C) @ (C, C) -> (1, C)

        return dLdZ
