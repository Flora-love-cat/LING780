"""
Word2Vec Exercises

For these exercises, fill out the following functions.
"""
from typing import List, Tuple

import numpy as np


def load_embeddings(filename: str = "word2vec_embeddings.txt") \
        -> Tuple[List[str], np.ndarray]:
    """
    Loads word2vec word embeddings from a .txt file.

    :param filename: The name of the file containing the embeddings
    :return: A vocabulary of words, and an array of shape (num_words,
        embedding_size) where each row is the word embedding for the
        corresponding word in the vocabulary
    """
    with open(filename, "r") as f:
        words, vecs = zip(*[line.split(" ", 1) for line in f])
    return list(words), np.array([np.fromstring(v, sep=" ") for v in vecs])


def get_embedding(w: str, vocab: List[str], all_embeds: np.ndarray) \
        -> np.ndarray:
    """
    Problem 16.

    Look up the embedding of a word from the matrix of word embeddings.

    :param w: A word
    :param vocab: A vocabulary
    :param all_embeds: A matrix of word embeddings, where all_embeds[i]
        is the embedding of vocab[i]
    :return: The word embedding for w
    """
    return all_embeds[vocab.index(w)]


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Problem 17.

    Compute the cosine similarity between two word embeddings or
    matrices of word embeddings.

    :param x: A 1-dimensional array of shape (embedding_size,)
        containing a single word embedding, or a 2-dimensional array of
        shape (m, embedding_size) where each row is a word embedding
    :param y: A 1-dimensional array of shape (embedding_size,)
        containing a single word embedding, or a 2-dimensional array of
        shape (n, embedding_size) where each row is a word embedding
    :return: An array containing the cosine distance between each row of
        x and each row of y. If x and y are both 1-dimensional, then the
        shape of this array should be (); i.e., it should be a scalar.
        Otherwise, the shape of this array should be (m, n).
    """
    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return x_norm @ y_norm.T


def get_neighbors_of_embedding(embedding: np.ndarray, k: int, vocab: List[str],
                               all_embeds: np.ndarray) -> List[str]:
    """
    Problem 18.

    Find the top k words with greatest cosine similarity to a given word
    embedding.

    :param embedding: A word embedding
    :param k: The number of words to return
    :param vocab: A vocabulary
    :param all_embeds: A matrix of word embeddings, where all_embeds[i]
        is the embedding of vocab[i]
    :return: The top k words from vocab and all_embeds whose embeddings
        are most similar to embedding
    """
    cosine_similarities = cosine_sim(all_embeds, embedding)
    top_k_indices = np.argpartition(-cosine_similarities, k)[:k]
    return [vocab[i] for i in top_k_indices]


def get_neighbors_of_word(word: str, k: int, vocab: List[str],
                          all_embeds: np.ndarray) -> List[str]:
    """
    Problem 19.

    Find the top k words with greatest cosine similarity to a given
    word.

    :param word: A word
    :param k: The number of words to return
    :param vocab: A vocabulary
    :param all_embeds: A matrix of word embeddings, where all_embeds[i]
        is the embedding of vocab[i]
    :return: The top k words from vocab and all_embeds whose embeddings
        are most similar to word
    """
    embedding = get_embedding(word, vocab, all_embeds)
    neighbors = get_neighbors_of_embedding(embedding, k, vocab, all_embeds)
    return [w for w in neighbors if w != word]


def analogy(w1: str, w2: str, w3: str, vocab: List[str],
            all_embeds: np.ndarray, k: int = 1) -> List[str]:
    """
    Problem 21.

    Find the top k candidates for the fourth word in an analogy.

    :param w1: A word
    :param w2: A word
    :param w3: A word
    :param vocab: A vocabulary
    :param all_embeds: A matrix of word embeddings, where all_embeds[i]
        is the embedding of vocab[i]
    :param k: The number of words to return
    :return: The top k words whose embeddings are most similar to
        [[w2]] - [[w1]] + [[w3]]
    """
    e1 = get_embedding(w1, vocab, all_embeds)
    e2 = get_embedding(w2, vocab, all_embeds)
    e3 = get_embedding(w3, vocab, all_embeds)
    return get_neighbors_of_embedding(e2 - e1 + e3, k, vocab, all_embeds)
