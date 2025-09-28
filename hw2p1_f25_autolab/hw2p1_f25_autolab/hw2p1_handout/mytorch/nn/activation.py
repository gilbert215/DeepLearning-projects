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
        self.A = np.tanh(Z)
        #self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
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