"""
Code for Problems 11, 12, 13, and 14.
"""
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from data_loader import Vocabulary


class MLPPosTagger(nn.Module):
    """
    MLP POS Tagger.
    """

    def __init__(self, token_vocab: Vocabulary, pos_tag_vocab: Vocabulary,
                 ngram_len: int, embedding_size: int, hidden_size: int,
                 num_hidden_layers: int = 0,
                 pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Problem 11: Complete the definition of the MLP POS tagging
        model. We have already implemented the embedding layer for you.
        You are responsible for implementing the output layer and hidden
        layers, if applicable.

        :param token_vocab: The input vocabulary, which contains all the
            possible tokens appearing in sentences along with the [UNK],
            [BOS], and [PAD] tokens
        :param pos_tag_vocab: The output vocabulary, which contains all
            the possible POS tags along with the [PAD] token
        :param ngram_len: The length of n-grams that the model will use
        :param embedding_size: The size of the embeddings that the model
            will use
        :param hidden_size: The output size of hidden layers, if
            num_hidden_layers > 0
        :param num_hidden_layers: The number of hidden layers of the MLP
            network
        :param pretrained_embeddings: A matrix of pre-trained GloVe or
            word2vec embeddings, if you wish to use them. Shape: (vocab
            size, embedding size)
        """
        if ngram_len % 2 == 0:
            raise ValueError("{} is not a valid n-gram length."
                             "".format(ngram_len))

        super().__init__()

        vocab_size = len(token_vocab)
        self.token_vocab = token_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.ngram_len = ngram_len
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.token_bos_index = token_vocab.get_index("[BOS]")
        self.token_pad_index = token_vocab.get_index("[PAD]")

        # Load pre-trained embeddings
        if pretrained_embeddings is not None:
            assert vocab_size == len(pretrained_embeddings) + 3
            assert embedding_size == pretrained_embeddings.shape[1]

            pretrained_embeddings = torch.cat([
                torch.Tensor(pretrained_embeddings),
                torch.randn(3, embedding_size)])
            self.embedding_layer = \
                nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # Problem 11: Replace the following line with your own code. Do
        # not edit anything above this line in this function.
        # self.layers: nn.Sequential = None
        layers = []
        for i in range(num_hidden_layers + 1):
            input_size = self.embedding_size * self.ngram_len if i == 0 else self.hidden_size
            output_size = len(self.pos_tag_vocab) if i == num_hidden_layers else self.hidden_size
            layers.append(nn.Linear(input_size, output_size))
            if i < num_hidden_layers:
                layers.append(nn.Tanh())
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, sentences: torch.LongTensor) -> torch.Tensor:
        """
        Problem 12: Implement the forward pass for the MLP model.

        :param sentences: A mini-batch of sentences, not including [BOS]
            but padded with [PAD], that will be labeled by the model.
            Shape: (batch size, max sentence length)
        :return: Logits for each of the tokens in sentences, where [PAD]
            is a valid prediction. Shape: (batch size,
            max sentence length, number of POS tags + 1)
        """
        # Insert your code here!

        input = torch.cat([torch.full((sentences.shape[0], self.ngram_len // 2), self.token_bos_index), 
           sentences,
          torch.full((sentences.shape[0], self.ngram_len // 2), self.token_pad_index)], 
          dim=-1) 
        
        # get n-gram indices
        ngrams = input.unfold(-1, self.ngram_len, 1) 

        # Convert the n-gram indices to embeddings
        embeddings = self.embedding_layer(ngrams)

        # Reshape the embeddings by concatenating them together
        layer_input = embeddings.reshape(shape=(embeddings.shape[0], embeddings.shape[1], -1))

        # Run the forward pass of all the layers
        return self.layers(layer_input)


class RNNPosTagger(nn.Module):
    """
    RNN POS tagger.
    """

    def __init__(self, token_vocab: Vocabulary, pos_tag_vocab: Vocabulary,
                 embedding_size: int, hidden_size: int,
                 num_rnn_layers: int = 1, bidirectional: bool = False,
                 rnn_type: str = "lstm",
                 pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Problem 13: Complete the definition of the RNN POS tagging
        model. We have already implemented the embedding layer for you.
        You are responsible for creating the RNN cell and the linear
        decoder.

        :param token_vocab: The vocabulary of inputs to the model
        :param pos_tag_vocab: The POS tag vocabulary, including [PAD]
        :param embedding_size: The size of the word embeddings used by
            the model
        :param hidden_size: The size of the RNN's hidden state vector
        :param num_rnn_layers: The number of RNN layers to use
        :param bidirectional: Whether or not the RNN is bidirectional
        :param rnn_type: The type of RNN to use. Possible values are
            "lstm", "gru", or "srn"
        :param pretrained_embeddings: A matrix of pre-trained GloVe or
            word2vec embeddings, if you wish to use them. Shape: (vocab
            size, embedding size)
        """
        super().__init__()
        vocab_size = len(token_vocab)
        num_pos_tags = len(pos_tag_vocab)
        self.token_vocab = token_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.token_pad_index = token_vocab.get_index("[PAD]")
        self.rnn_type = rnn_type
        self.rnn_hidden_states: Optional[torch.FloatTensor] = None

        # Load pre-trained embeddings
        if pretrained_embeddings is not None:
            assert vocab_size == len(pretrained_embeddings) + 3
            assert embedding_size == pretrained_embeddings.shape[1]

            pretrained_embeddings = torch.cat([
                torch.Tensor(pretrained_embeddings),
                torch.randn(3, embedding_size)])
            self.embedding_layer = \
                nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # Problem 13: Replace the following two lines with your own
        # code. Do not edit anything above this line in this function.
        self.rnn: Union[nn.RNN, nn.LSTM, nn.GRU] = None

        if rnn_type == "srn":

            self.rnn: Union[nn.RNN, nn.LSTM, nn.GRU] = nn.RNN(self.embedding_size, self.hidden_size, 
            num_layers=num_rnn_layers, nonlinearity='tanh', bias=True, 
            batch_first=True, dropout=0, bidirectional=bidirectional)

        elif rnn_type == "lstm":

            self.rnn: Union[nn.RNN, nn.LSTM, nn.GRU] = nn.LSTM(self.embedding_size, self.hidden_size, 
            num_layers=num_rnn_layers, bias=True, 
            batch_first=True, dropout=0, bidirectional=bidirectional, 
            proj_size=0)
            
        elif rnn_type == "gru":

            self.rnn: Union[nn.RNN, nn.LSTM, nn.GRU] = nn.GRU(self.embedding_size, self.hidden_size, 
            num_layers=num_rnn_layers, bias=True, 
            batch_first=True, dropout=0, bidirectional=bidirectional) 

        D = 2 if bidirectional == True else 1

        self.decoder = nn.Linear(D * self.hidden_size, num_pos_tags)


    def forward(self, sentences: torch.LongTensor) -> torch.Tensor:
        """
        Problem 14: Implement the forward pass for the RNN model.

        :param sentences: A mini-batch of sentences, not including [BOS]
            but padded with [PAD], that will be labeled by the model.
            Shape: (batch size, sentence length)
        :return: Logits for each of the tokens in sentences, where [PAD]
            is a valid prediction. Shape: (batch size,
            max sentence length, number of POS tags + 1)
        """
        # Replace the following with your own code. Please set the
        # variable self.rnn_hidden_states to the hidden states of the
        # RNN cell.
        embeddings = self.embedding_layer(sentences)
        self.rnn_hidden_states = self.rnn(embeddings)[0] 
        return self.decoder(self.rnn_hidden_states) 


