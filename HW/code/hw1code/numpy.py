"""
NumPy Exercises

For these exercises, fill out the following functions. Do not import any
additional libraries.
"""
import time
from typing import List

import numpy as np

"""
Part 1.3: Complex Operations.

Implement the following functions in one line of code using NumPy 
operations. Your one line of code must be at most 80 characters wide.
Replace the "pass" line with your own code. (In Python, "pass" does 
nothing.)

Use the type hints for each function to see what kinds of inputs the
function accepts and what kinds of outputs (if any) the function must
return. Read the docstrings to find out what each function must do.
"""


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Problem 9.

    Apply the sigmoid function to each item in an array.

    Example: On input
        [[-1., -0.5, 0.],
         [1., 0.5, 0.]],
    the output should be
        [[0.26894142, 0.37754067, 0.5],
         [0.73105858, 0.62245933, 0.5]].

    :param x: An array
    :return: The sigmoid of x
    """
    return 1. / (1. + np.exp(-x))


def zero_center(x: np.ndarray) -> np.ndarray:
    """
    Problem 10.

    Given an array of row vectors, subtract the mean value of each row
    from the row.

    Example: Given the input
        [[1., 2., 3.],
         [4., 5., 6.],
         [-1., 3., 4.]],
    the output should be
        [[-1., 0., 1.],
         [-1., 0., 1.],
         [-3., 1., 2.]].

    :param x: An array of shape (n, m), where each row is an m-
        dimensional row vector
    :return: An array of shape (n, m), where the mean of each row is
        subtracted from the array's original values
    """
    return x - x.mean(axis=-1, keepdims=True)


def even_rows(x: np.ndarray) -> np.ndarray:
    """
    Problem 11.

    Return the rows (i.e., items along axis 0) with an even index from
    an array.

    Example: The even rows of the matrix
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    are
        [[1, 2, 3],
         [7, 8, 9]].

    :param x: An array
    :return: The items of x where the index along axis 0 are even
    """
    return x[::2]


def mask(x: np.ndarray, mask_val: float, replace_with: float = -1.):
    """
    Problem 12.

    Mask all instances of a certain value from an array by replacing
    them with some default value. Do this in-place; i.e., by changing
    the value of x instead of producing a return value.

    Example: Define an array x as follows:
        >>> x = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
    To mask the value 2 from x, we run the code:
        >>> mask(x, 2)
    After running this function on x, the value of x is:
        >>> x
        array([[ 1, -1,  3],
               [-1,  1,  3],
               [ 3, -1,  1]])

    :param x: The array that the mask will be applied to
    :param mask_val: The value that will be masked out
    :param replace_with: The value to replace mask_val with
    """
    x[x == mask_val] = replace_with


def accuracy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Problem 13.

    In this function, you are given an array containing the output of a
    multi-class classifier, along with an array containing the correct
    classifications (or "labels"). You must compute the accuracy of the
    classifier's predictions.

    Example: Suppose a 3-class classifier produces the output
        [[1.5, -.75, .25],
         [-2.1, -1.3, -.5],
         [-.1, 2.5, 1.4],
         [.3, -.01, .15]].
    Each row contains the classifier's prediction on some example.
    Column 0 contains a confidence score, or "logit score," describing
    the degree to which the classifier believes the example belongs to
    Class 0. Similarly column 1 contains the logit score for Class 1,
    and column 2 contains the logit score for Class 2. The classifier's
    prediction is the class with the highest logit score.

    If the correct classifications for the for examples above are
        [0, 2, 1, 2],
    then the classifier achieves an accuracy of .75, since the first
    three predictions are correct out of a total of four examples.

    :param logits: An array of shape (n, m) containing logit scores
        computed by an m-class classifier for n examples
    :param labels: An array of shape (n,) containing the correct
        classifications for the n examples
    :return: The proportion of examples for which the highest logit
        score in logits matches the classification given by labels
    """
    return (logits.argmax(axis=-1) == labels).sum() / len(labels)


"""
Part 1.4: Analysis of Matrix Multiplication.

In NumPy, matrices are represented as 2-dimensional Arrays. Without 
NumPy, matrices are represented as lists of lists of numbers. For
example, the list [[1., 2., 3], [4., 5., 6.]] represents a 2-by-3 matrix
where the first row contains 1, 2, and 3, and the second row contains
4, 5, and 6.

In this part of the assignment, you will compare the performance of 
NumPy arrays against lists of lists of numbers. First, fill in the 
matmul_pure_python function with your own implementation of matrix
multiplication. Then, run the measure_time function to compare the
running time of your function with NumPy's @ operator. Record and turn
in the two running times you have measured.

Which implementation of matrix multiplication is faster: the pure Python
version or the NumPy version?
"""

Matrix = List[List[float]]  # This is called a "type alias"


def matmul_pure_python(a: Matrix, b: Matrix) -> Matrix:
    """
    Problem 14.
    
    Multiplies two matrices using only Python built-in functions. Do not
    use NumPy or any other external library.

    :param a: A matrix
    :param b: Another matrix
    :return: a @ b, the product of a and b under matrix multiplication
    """
    num_rows = len(a)  # Number of rows of a
    num_cols = len(b[0])  # Number of columns of b
    inner_dim = len(b)  # Number of rows of b

    return [[sum(a[i][k] * b[k][j] for k in range(inner_dim))
             for j in range(num_cols)] for i in range(num_rows)]


def matmul_numpy(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiplies two matrices using NumPy.

    :param a: A matrix
    :param b: Another matrix
    :return: a @ b, the product of a and b under matrix multiplication
    """
    return (np.array(a) @ np.array(b)).tolist()


def measure_time():
    """
    Run this function to measure the performance of matmul_pure_python
    against NumPy's implementation of matrix multiplication.
    """
    # Generate some random matrices
    a = np.random.rand(200, 300).tolist()
    b = np.random.rand(300, 400).tolist()

    # Record time for matmul_pure_python
    start_time = time.time()
    _ = matmul_pure_python(a, b)  # _ is a dummy variable
    end_time = time.time()

    elapsed = end_time - start_time
    print("Matrix multiplication in pure Python took {:.3f} "
          "seconds.".format(elapsed))

    # Record time for matmul_numpy
    start_time = time.time()
    _ = matmul_numpy(a, b)
    end_time = time.time()

    elapsed = end_time - start_time
    print("Matrix multiplication in NumPy took {:.3f} "
          "seconds.".format(elapsed))


# The if __name__ == "__main__": statement contains code that is run
# when a Python file is executed as a module. You can use this part of
# the file to test your code and run the measure_time function. We will
# not grade this part of the code.
if __name__ == "__main__":
    measure_time()
