import pickle
from typing import Optional, Tuple

import numpy as np

from data_loader import load_embeddings, all_pos_tags, Dataset, Vocabulary
from loss import CrossEntropyLoss
from metrics import AverageLoss, Accuracy
from model import MultiLayerPerceptron


def print_metrics(loss: float, accuracy: float, message: str):
    print("{}. Loss: {:.2f}, Accuracy: {:.3f}".format(message, loss, accuracy))


def train_epoch(model: MultiLayerPerceptron, train_data: Dataset,
                batch_size: int, loss_function: CrossEntropyLoss, lr: float):
    """
    Trains a model for one epoch.

    :param model: A model
    :param train_data: The training data
    :param batch_size: The batch size
    :param loss_function: The loss function layer
    :param lr: The learning rate for this batch
    """
    loss_metric = AverageLoss()
    acc_metric = Accuracy()

    for i, (ngrams, pos_tags) in enumerate(train_data.get_batches(batch_size)):
        # implement one iteration of the SGD algorithm. 
        # `ngrams` contains the input to your model, 
        # `pos_tags` contains the gold-standard POS tags that model needs to predict.
        
        # clear the gradients of model
        model.clear_grad()
        # Set this variable to the output of your model, excluding softmax
        logits = model(ngrams)
        # compute total cross-entropy loss for this mini-batch
        batch_loss = loss_function(logits, pos_tags)
        # run backpropagation algorithm to compute gradients of model for this mini-batch
        model.backward(loss_function.backward())
        # update the parameters
        model.update(lr)

        # Update metrics (loss and accuracy)
        avg_batch_loss = loss_metric.update(batch_loss, len(ngrams))
        batch_acc = acc_metric.update(logits, pos_tags)
        if (i + 1) % 100 == 0:
            print_metrics(avg_batch_loss, batch_acc, "Batch {}".format(i + 1))

    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, "Training Complete")


def test(model: MultiLayerPerceptron, test_data: Dataset,
         loss_function: CrossEntropyLoss, message: str) -> Tuple[float, float]:
    """
    Evaluates a model using a dev set or test set.

    :param model: A model
    :param test_data: The data to evaluate the model on
    :param loss_function: The loss function layer
    :param message: A message to be displayed when showing results
    :return: The average loss and accuracy attained by model on
        test_data
    """
    loss_metric = AverageLoss()
    acc_metric = Accuracy()

    for i, (ngrams, pos_tags) in enumerate(test_data.get_batches(100)):
        # Forward pass
        logits = model(ngrams)
        batch_loss = loss_function(logits, pos_tags)

        # Update metrics
        _ = loss_metric.update(batch_loss, len(ngrams))
        _ = acc_metric.update(logits, pos_tags)

    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, message)
    return loss_metric.value, acc_metric.value


def run_trial(train_data: Dataset, dev_data: Dataset, test_data: Dataset,
              token_vocab: Vocabulary, pos_tag_vocab: Vocabulary,
              pretrained_embeddings: Optional[np.array] = None,
              filename: str = "checkpoint.p", ngram_len: int = 3,
              hidden_size: int = 100, num_layers: int = 1,
              num_epochs: int = 10, batch_size: int = 64, lr: float = 1.,
              step_size: int = 5, gamma: float = 1.):
    """
    Creates and trains a model with one particular configuration of
    hyperparameters.

    DATASET HYPERPARAMETERS:

    :param train_data: The training data
    :param dev_data: The validation data
    :param test_data: The testing data
    :param token_vocab: The Vocabulary for tokens
    :param pos_tag_vocab: The Vocabulary for POS tags

    MODEL HYPERPARAMETERS:

    :param pretrained_embeddings: A matrix of pre-trained word2vec
        embeddings, used to initialize the embedding matrix. If this is
        set to None, then the embedding matrix is initialized randomly
    :param filename: The filename to save checkpoints to. It should end
        with ".p"
    :param ngram_len: The size of the n-grams
    :param hidden_size: The size of the model's hidden layer, if it has
        more than 1 layer
    :param num_layers: The number of layers in the multi-layer
        perceptron

    TRAINING LOOP HYPERPARAMETERS:

    :param num_epochs: The number of epochs to train the model for
    :param batch_size: The size of the mini-batches used to train the
        model

    SGD AND LEARNING RATE SCHEDULING HYPERPARAMETERS:

    :param lr: The initial learning rate used for SGD
    :param step_size: The frequency (in number of epochs) with which the
        learning rate is annealed
    :param gamma: The decay factor by which the learning rate is
        multiplied when it is annealed
    """
    # Create model: with pretrained embedding
    model = MultiLayerPerceptron(len(token_vocab), len(pos_tag_vocab),
                                 ngram_len, pretrained_embeddings.shape[-1],
                                 hidden_size, num_layers)

    # Create model with no pretrained embedding
    # model = MultiLayerPerceptron(len(token_vocab), len(pos_tag_vocab),
    #                              ngram_len, 100,
    #                              hidden_size, num_layers)                             

    if pretrained_embeddings is not None:
        embedding_matrix = model.embedding_layer.params["embeddings"]
        embedding_matrix[:len(pretrained_embeddings)] = pretrained_embeddings

    loss_function = CrossEntropyLoss()

    # Train
    best_dev_acc = 0.
    best_epoch = None
    for i in range(num_epochs):
        print("Epoch {}".format(i))

        # Anneal learning rate
        if i > 0 and i % step_size == 0:
            lr *= gamma
            print("Annealing learning rate to {}.".format(lr))

        # Train model
        train_epoch(model, train_data, batch_size, loss_function, lr)
        _, dev_acc = test(model, dev_data, loss_function,
                          "Epoch {} Validation".format(i))

        # Save checkpoint
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = i

            print("Saving checkpoint...")
            with open(filename, "wb") as f:
                pickle.dump(model, f)
            print("Done.")

    print("The best validation accuracy occurred after epoch {}."
          "".format(best_epoch))
    with open(filename, "rb") as f:
        model = pickle.load(f)
    _ = test(model, test_data, loss_function, "Test")


if __name__ == "__main__":
    # Load embeddings and vocab
    print("Loading embeddings...")
    all_tokens, glove_embeddings = load_embeddings()

    # Only use the 5,000 most common embeddings
    all_tokens = all_tokens[:5000]
    glove_embeddings = glove_embeddings[:5000]
    print("Done.")

    token_vocab = Vocabulary(all_tokens + ["[UNK]", "[BOS]", "[EOS]"])
    pos_tag_vocab = Vocabulary(all_pos_tags)

    # Load training data
    print("Loading data...")
    train_data = Dataset.from_conll("data/en_ewt-ud-train.conllu", token_vocab,
                                    pos_tag_vocab)
    dev_data = Dataset.from_conll("data/en_ewt-ud-dev.conllu", token_vocab,
                                  pos_tag_vocab)
    test_data = Dataset.from_conll("data/en_ewt-ud-test.conllu", token_vocab,
                                   pos_tag_vocab)
    print("Done.")

    # Train
    run_trial(train_data, dev_data, test_data, token_vocab, pos_tag_vocab,
              glove_embeddings, "checkpoint.p", lr=.1, num_epochs=4,
              num_layers=1, step_size=2, gamma=.25, batch_size=5,
              hidden_size=30)
