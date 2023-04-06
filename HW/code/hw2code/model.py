from typing import List

import numpy as np

from layers import Embedding, Layer, Linear, Tanh


class MultiLayerPerceptron(Layer):
    """
    A multi-layer perceptron. This class is a wrapper around the layers
    comprising a full multi-layer perceptron model.
    """

    def __init__(self, vocab_size: int, num_pos_tags: int, ngram_len: int,
                 embedding_size: int, hidden_size: int, num_layers: int):
        """
        Initializes a MultiLayerPerceptron from a list of existing
        Layers. The softmax activation function is not included.

        :param vocab_size: The size of the token vocabulary
        :param num_pos_tags: The number of distinct POS tags in the POS
            tag vocabulary
        :param ngram_len: The size of the n-grams
        :param embedding_size: The size of word embeddings used by the
            model
        :param hidden_size: The size of hidden layer representations
            used by the model (only relevant if the model has more than
            1 layer)
        :param num_layers: The number of layers in the model
        """
        super().__init__()

        # Save basic hyperparameters of the model
        self.vocab_size = vocab_size
        self.num_pos_tags = num_pos_tags
        self.ngram_len = ngram_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Construct layers
        self.embedding_layer: Embedding = Embedding(embedding_size, vocab_size)
        self.layers: List[Layer] = []

        for i in range(num_layers):
            input_size = embedding_size * ngram_len if i == 0 else hidden_size
            output_size = num_pos_tags if i == num_layers - 1 else hidden_size
            self.layers.append(Linear(input_size, output_size))
            if i < num_layers - 1:
                self.layers.append(Tanh())

    def clear_grad(self):
        """
        Clears the gradients of all layers.
        """
        self.embedding_layer.clear_grad()
        for layer in self.layers:
            layer.clear_grad()

    def forward(self, ngrams: np.ndarray) -> np.ndarray:
        """
        Applies the forward pass to all layers, in order.

        :param ngrams: An int-valued array of vocabulary indices. Shape:
            (batch size, n-gram length)
        :return: An array of logit scores. The softmax of this array
            gives the probability assigned to each POS tag. Shape:
            (batch size, number of POS tags)
        """
        # Convert the n-gram indices to embeddings
        embeddings = self.embedding_layer(ngrams)

        # Reshape the embeddings by concatenating them together
        layer_input = embeddings.reshape((len(ngrams), -1))

        # Run the forward pass of all the layers
        for layer in self.layers:
            layer_input = layer(layer_input)

        return layer_input

    def backward(self, delta: np.ndarray):
        """
        Problem 6: Implement the backward pass. This function needs to
        call the backward function of self.embedding_layer and all
        layers in self.layers. It does not need to return anything.

        :param delta: The gradient of the objective with respect to the
            logit scores produced by self.forward. Shape: (batch size,
            number of POS tags)
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        self.embedding_layer.backward(delta.reshape(-1,self.ngram_len,self.embedding_size))


        

    def update(self, lr: float):
        """
        Problem 8: Implement the SGD update for MultiLayerPerceptron.
        This function must update the parameters of self.embedding_layer
        as well as all the layers in self.layers.

        :param lr: The learning rate
        """
        self.embedding_layer.update(lr)

        for layer in (self.layers):
            layer.update(lr)
