"""
Code for loading data and abstractions for representing data. DO NOT
EDIT THIS FILE.
"""
from typing import Generator, List, Tuple

import numpy as np
import pyconll
import torch

from _utils import cache_pickle, timer

all_pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
                "VERB", "X"]


@cache_pickle
def load_embeddings(filename: str = "data/glove_embeddings.txt") \
        -> Tuple[List[str], np.ndarray]:
    """
    Loads word embeddings from a .txt file.

    :param filename: The name of the file containing the embeddings
    :param cache_filename: If this keyword argument is provided, the
        embeddings will be cached to a pickle file whose filename is
        given by this keyword argument
    :return: A vocabulary of words, and an array of shape (num_words,
        embedding_size) where each row is the word embedding for the
        corresponding word in the vocabulary
    """
    with timer("Loading pre-trained embeddings from {}...".format(filename)):
        with open(filename, "r") as f:
            words, vecs = zip(*[line.split(" ", 1) for line in f])
        words = list(words)
        vecs = np.array([np.fromstring(v, sep=" ") for v in vecs])
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

    def __init__(self, sentences: List[List[int]], pos_tags: List[List[int]],
                 token_pad_index: int, pos_tag_pad_index: int,
                 sort_by_length: bool = True):
        """
        Creates a dataset from pre-processed data. Data should be in the
        form of lists of lists of indices.

        :param sentences: The sentences that need to be POS-tagged.
        :param pos_tags: The POS tags corresponding to the sentences.
        :param token_pad_index: The index of the [PAD] token within the
            token vocabulary.
        :param pos_tag_pad_index: The index of the [PAD] token within
            the POS tag vocabulary.
        :param sort_by_length: If True, the dataset will be sorted by
            length
        """
        self.token_pad_index = token_pad_index
        self.pos_tag_pad_index = pos_tag_pad_index

        # Sort data by length
        if sort_by_length:
            all_data = list(zip(sentences, pos_tags))
            all_data.sort(key=lambda x: len(x[0]))
            self._sentences, self._pos_tags = zip(*all_data)
        else:
            self._sentences = sentences
            self._pos_tags = pos_tags

    def __len__(self):
        return len(self._sentences)

    @property
    def labels(self):
        return self._pos_tags

    def get_batches(self, batch_size: int) \
            -> Generator[Tuple[torch.LongTensor, torch.LongTensor], None,
                         None]:
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

            # Pad the batch
            seq_len = max(len(s) for s in self._sentences[i:j])

            sentences = [s + [self.token_pad_index] * (seq_len - len(s))
                         for s in self._sentences[i:j]]
            pos_tags = [pt + [self.pos_tag_pad_index] * (seq_len - len(pt))
                        for pt in self._pos_tags[i:j]]

            yield torch.LongTensor(sentences), torch.LongTensor(pos_tags)

    @classmethod
    @cache_pickle
    def from_conll(cls, raw_data_filename: str, token_vocab: Vocabulary,
                   pos_tag_vocab: Vocabulary):
        """
        Creates a Dataset from a CoNLL file. This function reads the
        data from the CoNLL file and processes the data into index
        lists.

        :param raw_data_filename: The name of the CoNLL file
        :param token_vocab: A Vocabulary for the tokens
        :param pos_tag_vocab: A Vocabulary for the POS tags
        :param cache_filename: If this keyword argument is provided, the
            embeddings will be cached to a pickle file whose filename is
            given by this keyword argument
        :return: A Dataset contatining the data in the CoNLL file
        """
        raw_data = pyconll.load_from_file(raw_data_filename)

        token_idx: List[List[int]] = []
        pos_tag_idx: List[List[int]] = []
        for sentence in raw_data:
            # Extract tokens and tags
            tokens, pos_tags = zip(*[(t.form, t.upos) for t in sentence])

            # Create token indices
            tokens = [t if t in token_vocab else "[UNK]" for t in tokens]
            token_idx.append([token_vocab.get_index(t) for t in tokens])

            # Create POS tag indices
            pos_tags = [p if p in pos_tag_vocab else "X" for p in
                        pos_tags]
            pos_tag_idx.append([pos_tag_vocab.get_index(p) for p in pos_tags])

        return cls(token_idx, pos_tag_idx, token_vocab.get_index("[PAD]"),
                   pos_tag_vocab.get_index("[PAD]"))
