import numpy as np

from layers import Layer


def softmax(x: np.array) -> np.array:
    """
    Problem 9: Implement a numerically stabilized version of softmax.

    :param x: Any array of vectors
    :return: The softmax of x - c along dimension -1, where c is the
        maximum of x along dimension -1
    """
    # pass  # Replace this with your own code
    c = np.max(x, axis=-1, keepdims=True)
    return np.exp(x-c) / np.exp(x-c).sum(axis=-1, keepdims=True)


class CrossEntropyLoss(Layer):
    """
    This class computes the cross-entropy loss for an array of logit
    scores, to which softmax has not been applied.
    """

    def __init__(self):
        """
        This layer has no parameters. It is up to you whether you want
        to put anything else in this function.
        """
        super().__init__()
        self.logits = None 
        self.y = None 

    def __call__(self, logits: np.ndarray, y: np.ndarray) -> float:
        return self.forward(logits, y)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Problem 11: Compute the total cross-entropy loss for a batch of
        neural network predictions given a set of gold labels. You may
        add code to the __init__ method if you would like.

        :param logits: An array of logit scores. The softmax of this
            array gives the probability assigned to each POS tag. Shape:
            (batch size, number of POS tags)
        :param y: An int-valued array of gold-standard POS tag labels.
            Shape: (batch size)
        :return: A 0D (scalar) array containing the sum of the cross-
            entropy loss for all predictions in the batch
        """
        self.logits = logits 
        self.y = y 
        return -np.sum(np.log(softmax(logits)[np.arange(logits.shape[0]), y])) 

    def backward(self) -> np.ndarray:
        """
        Problem 12: Computes the gradient of the cross-entropy loss with
        respect to the logit scores. Since this is the root node of the
        computation graph, the backward function does not take any
        parameters. You may add code to the __init__ method if you would
        like.

        :return: The gradient of the cross-entropy loss with respect to
            the logit scores. Shape: (batch size, number of POS tags)
        """
        one_hot = np.zeros(self.logits.shape)
        one_hot[np.arange(self.logits.shape[0]), self.y] = 1
        return softmax(self.logits) - one_hot
