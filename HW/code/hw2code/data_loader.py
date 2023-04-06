import os.path
import pickle
from typing import Generator, List, Tuple

import numpy as np
import pyconll

all_pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
                "VERB", "X"]


def load_embeddings(filename: str = "data/glove_embeddings.txt") \
        -> Tuple[List[str], np.ndarray]:
    """
    Loads word embeddings from a .txt file.

    :param filename: The name of the file containing the embeddings
    :return: A vocabulary of words, and an array of shape (num_words,
        embedding_size) where each row is the word embedding for the
        corresponding word in the vocabulary
    """
    if os.path.exists("{}.p".format(filename)):
        with open("{}.p".format(filename), "rb") as f:
            vector_dict = pickle.load(f)
        words = vector_dict["words"]
        vecs = vector_dict["vecs"]
    else:
        with open(filename, "r") as f:
            words, vecs = zip(*[line.split(" ", 1) for line in f])
        words = list(words)
        vecs = np.array([np.fromstring(v, sep=" ") for v in vecs])
        with open("{}.p".format(filename), "wb") as f:
            pickle.dump({"words": words, "vecs": vecs}, f)
    return words, vecs


class Vocabulary(object):
    """
    A container that maintains a mapping between tokens or POS tags and
    their indices.
    """

    def __init__(self, forms: List[str]):
        self.forms = forms
        self.indices = {j: i for i, j in enumerate(forms)}

    def get_index(self, form: str) -> int:
        """
        Looks up the index of a token or POS tag.

        :param form: The string form of the token or POS tag
        :return: The index for the token or POS tag
        """
        return self.indices[form]

    def get_form(self, index: int) -> str:
        """
        Looks up the string form of a token or POS tag.

        :param index: The index for the token or POS tag
        :return: The string form of the token or POS tag
        """
        return self.forms[index]

    def get_ngrams(self, tokens: List[str], ngram_len: int) -> List[List[int]]:
        """
        Converts a list of tokens into n-grams.

        :param tokens: A list of tokens
        :param ngram_len: The size of the n-grams
        :return: The n-grams from tokens, represented as indices
        """
        if ngram_len % 2 == 0:
            raise ValueError("ngram_len must be odd!")
        k = int((ngram_len - 1) / 2)

        tokens = [t if t in self else "[UNK]" for t in tokens]
        tokens = ["[BOS]"] * k + tokens + ["[EOS]"] * k
        idx = [self.get_index(t) for t in tokens]
        return [idx[i - k:i + k + 1] for i in range(k, len(tokens) - k)]

    def __contains__(self, item: str) -> bool:
        return item in self.forms

    def __len__(self) -> int:
        return len(self.forms)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.forms)


class Dataset(object):
    """
    A container that stores a dataset and divides it into batches.
    """

    def __init__(self, ngrams: np.ndarray, pos_tags: np.ndarray):
        """
        Creates a dataset from pre-processed data. All data should
        already be in the form of int-valued index arrays.

        :param ngrams: The n-grams of the data. Shape: (number of
            examples, n-gram length)
        :param pos_tags: The POS tags corresponding to the n-grams.
            Shape: (number of examples,)
        """
        self.ngrams = np.array(ngrams)
        self.pos_tags = np.array(pos_tags)

    def __len__(self):
        return len(self.ngrams)

    def get_batches(self, batch_size: int) \
            -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Creates a generator that loops over the data in the form of
        mini-batches.

        :param batch_size: The size of each mini-batch
        :return: Each item in the generator should contain an array of
            n-grams of shape (batch size, n-gram length) and an array of
            POS tags of shape (batch size,).
        """
        for i in range(0, len(self), batch_size):
            j = i + batch_size
            yield self.ngrams[i:j], self.pos_tags[i:j]

    @classmethod
    def from_conll(cls, filename: str, token_vocab: Vocabulary,
                   pos_tag_vocab: Vocabulary, ngram_len: int = 3):
        """
        Creates a Dataset from a CoNLL file. This function reads the
        data from the CoNLL file and processes the data into index
        arrays.

        :param filename: The name of the CoNLL file
        :param token_vocab: A Vocabulary for the tokens
        :param pos_tag_vocab: A Vocabulary for the POS tags
        :param ngram_len: The size of the n-grams
        :return: A Dataset contatining the data in the CoNLL file
        """
        if os.path.exists("{}-{}gram.p".format(filename, ngram_len)):
            with open("{}-{}gram.p".format(filename, ngram_len), "rb") as f:
                data_dict = pickle.load(f)
            ngram_idx_array: np.ndarray = data_dict["ngram_idx"]
            pos_tag_idx_array: np.ndarray = data_dict["pos_tag_idx"]
        else:
            raw_data = pyconll.load_from_file(filename)

            ngram_idx: List[List[int]] = []
            pos_tag_idx: List[int] = []
            for sentence in raw_data:
                # Extract tokens and tags
                tokens, pos_tags = zip(*[(t.form, t.upos) for t in sentence])

                # Create n-gram indices
                ngram_idx += token_vocab.get_ngrams(tokens, ngram_len)

                # Create POS tag indices
                pos_tags = [pt if pt in pos_tag_vocab else "X" for pt in
                            pos_tags]
                pos_tag_idx += [pos_tag_vocab.get_index(pt) for pt in pos_tags]

            ngram_idx_array = np.array(ngram_idx)
            pos_tag_idx_array = np.array(pos_tag_idx)

            with open("{}-{}gram.p".format(filename, ngram_len), "wb") as f:
                pickle.dump({"ngram_idx": ngram_idx_array,
                             "pos_tag_idx": pos_tag_idx_array}, f)

        return cls(ngram_idx_array, pos_tag_idx_array)
