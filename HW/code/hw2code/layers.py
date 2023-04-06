import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class Layer(ABC):
    """
    A basic interface ("template" class) for a neural network layer. All
    neural network layer types should be subclasses of this class.
    """

    def __init__(self):
        # The layer parameters and gradients are stored in two dicts:
        # self.params and self.grad. These two dicts must have the same
        # keys, each of which represents the name of a parameter. For
        # any parameter p, self.params[p] must have the same shape as
        # self.grad[p]
        self.params: Dict[str, np.ndarray] = dict()
        self.grad: Dict[str, np.ndarray] = dict()

        # For backward, we need to save the input to forward
        self.x: Optional[np.ndarray] = None

    def clear_grad(self):
        """
        Sets all gradients to 0.
        """
        self.grad = {p: np.zeros(v.shape) for p, v in self.params.items()}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        This is a wrapper around self.forward, which saves the input as
        self.x. Always call forward using this wrapper!

        :param x: The layer input
        :return: The layer output value
        """
        self.x = x
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass. Each layer type needs to implement this.

        :param x: The layer input
        :return: The layer output value
        """
        pass

    @abstractmethod
    def backward(self, delta: np.ndarray) -> Optional[np.ndarray]:
        """
        The backward pass. Each layer type needs to implement this. The
        return value of this function should be the gradient with
        respect to self.x. Additionally, the gradient with respect to
        all parameters should be added to the appropriate arrays in
        self.grad.

        :param delta: The gradient of the objective with respect to the
            output value of this layer
        :return: The gradient of the objective with respect to self.x
        """
        pass 

    def update(self, lr: float):
        """
        Problem 7: Update the parameters in self.params according to the
        gradients in self.grad and a given learning rate.

        :param lr: The learning rate
        """
        for p in self.params:
            self.params[p] -= lr * self.grad[p]
        


class Embedding(Layer):
    """
    An embedding layer.
    """

    def __init__(self, embedding_size: int, vocab_size: int):
        """
        Initializes a random matrix of word embeddings.

        :param embedding_size: The dimensionality of the word embeddings
        :param vocab_size: The number of distinct word embeddings
            included in the word embedding matrix
        """
        super().__init__()

        # Initialize parameters and gradients
        self.params = {"embeddings": np.random.uniform(size=(vocab_size,
                                                             embedding_size))}
        self.clear_grad()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Looks up an array of vocabulary indices from the word embedding
        matrix.

        For example, if the input is
            x = np.array([[1, 2, 3],
                          [2, 3, 4]]),
        then this function should return a 3D array of shape (2, 3,
        embedding size) where
            self.forward(x)[0, 0] == self.params["embeddings"][1]
            self.forward(x)[0, 1] == self.params["embeddings"][2]
            self.forward(x)[0, 2] == self.params["embeddings"][3]
            self.forward(x)[1, 0] == self.params["embeddings"][2]
            self.forward(x)[1, 1] == self.params["embeddings"][3]
            self.forward(x)[1, 2] == self.params["embeddings"][4]

        :param x: An int-valued array of vocabulary indices. Shape:
            (batch size, n-gram length)
        :return: An array of embeddings, where each index in x has been
            replaced by its embedding. Shape: (batch size, n-gram
            length, embedding size)
        """
        return self.params["embeddings"][x]

    def backward(self, delta: np.ndarray):
        """
        Adds the gradients of the objective with respect to the layer
        output to the appropriate rows of self.grad["embeddings"].

        For example, if the input is
            x = np.array([[1, 2, 3],
                          [2, 3, 4]]),
        then delta is an array of shape (2, 3, embedding size)
        containing the gradient of the objective with respect to
        the word embeddings for vocabulary items 1, 2, 3, 2, 3, and 4.
        The gradients should then be updated as follows:
            self.grad["embeddings"][1] += delta[0, 0]
            self.grad["embeddings"][2] += delta[0, 1] + delta[1, 0]
            self.grad["embeddings"][3] += delta[0, 2] + delta[1, 1]
            self.grad["embeddings"][4] += delta[1, 2]

        :param delta: The gradient of the objective with respect to the
            word embedding vectors looked up from the embedding matrix.
            Shape: (batch size, n-gram length, embedding size)
        :return: None, since this layer is not differentiable
        """
        # Sort the vocabulary indices appearing in x
        x_argsorted = np.unravel_index(self.x.argsort(axis=None), self.x.shape)
        x_sorted = self.x[x_argsorted]

        # Build an array marking the places where x_sorted changes
        # value. For example, if x_sorted == [1, 1, 2, 2, 2, 3], then
        # groups == [0, 2, 5]
        groups = np.flatnonzero(np.concatenate([[1], np.diff(x_sorted)]))

        # Add up all the entries of delta corresponding to the same
        # vocabulary index given by x
        delta = np.add.reduceat(delta[x_argsorted], groups)

        # Update the corresponding entries of self.grad["embeddings"]
        self.grad["embeddings"][np.unique(x_sorted)] += delta


class Linear(Layer):
    """
    A linear layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Intializes a random weight matrix and bias vector.

        :param input_size: The dimensionality of the input vectors
        :param output_size: The dimensionality of the output vectors
        """
        super().__init__()

        # Initialize parameters and gradients
        k = math.sqrt(6 / (input_size + output_size))
        self.params = {"w": np.random.uniform(low=-k, high=k,
                                              size=(output_size, input_size)),
                       "b": np.zeros(output_size)}
        self.clear_grad()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Problem 3: Apply the linear map given by self.params. The input
        to this layer is a batch of vectors, arranged into a 3D array.
        This function must apply the linear map to each vector in the
        batch.

        :param x: A batch of input vectors. Shape: (batch size, input
            size)
        :return: The result of applying the linear map implemented by
            this layer to x. Shape: (batch size, output size)
        """
        return x @ (self.params['w']).T + self.params['b']

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Problem 5: Implement the backward pass for the linear layer.
        This function must return the gradient of the objective with
        respect to self.x and update the gradients in self.grad.

        :param delta: The gradient of the objective with respect to the
            output of this layer. Shape: (batch size, output size)
        :return: The gradient of the objective with respect to the input
            of this layer. Shape: (batch size, input size)
        """
        self.grad["w"] += delta.T @ self.x 
        self.grad["b"] += delta.T @ np.ones([delta.shape[0],])
        return delta @ self.params['w']

class Tanh(Layer):
    """
    The tanh activation function.
    """

    def __init__(self):
        """
        This layer has no parameters. It is up to you whether you want
        to put anything else in this function.
        """
        super().__init__()
        # You may add additional code here for Problems 2 and 4 if you
        # would like.
        self.x = None 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Problem 2: Apply the tanh function to the input. You may add
        code to the __init__ method if you would like.

        :param x: An input to tanh. Shape: any shape
        :return: The tanh of x. Shape: the same shape as x
        """
        self.x = x 
        return np.tanh(x) 

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Problem 4: Implement the backward pass for the tanh function.
        Since this layer has no parameters, self.grad does not need to
        be updated. You may add code to the __init__ method if you would
        like.

        :param delta: The gradient of the objective with respect to this
            layer's output. Shape: the same shape as self.x
        :return: The gradient of the objective with respect to self.x.
            Shape: the same shape as self.x
        """
        return delta * (1- (np.tanh(self.x))**2)
