"""
Code for Problems 15 and 16.
"""
from typing import List, Tuple, Union
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

from _utils import timer
from analysis import print_cm
from data_loader import Dataset
from metrics import AverageLoss, Accuracy
from models import MLPPosTagger, RNNPosTagger


def print_metrics(loss: float, accuracy: float, message: str):
    print("{}. Loss: {:.2f}, Accuracy: {:.3f}".format(message, loss, accuracy))


def train_epoch(model: Union[MLPPosTagger, RNNPosTagger], train_data: Dataset,
                batch_size: int, loss_function: nn.CrossEntropyLoss,
                optimizer: optim.Adam, pos_tag_pad_index: int):
    """
    Problem 15: Please complete this function, which trains a POS tagger
    for one epoch. Your code needs to be compatible with both the MLP
    and the RNN models.

    :param model: The model that will be trained
    :param train_data: The training data
    :param batch_size: The batch size
    :param loss_function: The loss function
    :param optimizer: The Adam optimizer

    DO NOT USE THIS PARAMETER:

    :param pos_tag_pad_index: The index of the [PAD] token in the output
        vocabulary
    """
    model.train()  # This needs to be called before the model is trained

    loss_metric = AverageLoss()
    acc_metric = Accuracy(pos_tag_pad_index)

    print("pos_tag_pad_index", pos_tag_pad_index)

    batches = train_data.get_batches(batch_size)
    for i, (sentences, pos_tags) in enumerate(batches):
        # Problem 15: Replace the following two lines with your code.
        # The variable batch_loss must contain the loss incurred for the current mini-batch. 
        # The variable logits must contain the output of model
        
        # set all the gradients to 0
        optimizer.zero_grad()
        # sentence (batch_size, seq_length) pos_tag (batch_size, sentence_length)
        output = model(sentences)   # output (batch_size, seq_length, num_classes)
        logits = output.reshape(-1, output.shape[2])   # (batch_size * seq_length, num_classes)
        pos_tags = pos_tags.flatten()  # (batch_size * sentence_length, )   an entry is a class index
        batch_loss = loss_function(logits, pos_tags)

        # backpropagation
        batch_loss.backward()
        
        # update params
        optimizer.step()
        
        # Update metrics. 
        avg_batch_loss = loss_metric.update(batch_loss, len(sentences))
        batch_acc = acc_metric.update(logits, pos_tags)
        if (i + 1) % 100 == 0:
            print_metrics(avg_batch_loss, batch_acc, "Batch {}".format(i + 1))

    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, "Training Complete")


def evaluate(model: Union[MLPPosTagger, RNNPosTagger], test_data: Dataset,
             loss_function: nn.CrossEntropyLoss, message: str,
             pos_tag_pad_index: int, print_conf_matrix: bool = False) \
        -> Tuple[float, float]:
    """
    Problem 16: Please complete this function, which evaluates a model
    using a training or development set.

    :param model: The model that will be evaluated
    :param test_data: The data to evaluate the model on
    :param loss_function: The loss function
    :param print_conf_matrix: Prints out a confusion matrix if True

    DO NOT USE THESE PARAMETERS:

    :param message: A message to be displayed when showing results
    :param pos_tag_pad_index: POS tag index to be ignored in evaluation

    :return: The average loss and accuracy attained by model on
        test_data
    """
    model.eval()  # This needs to be called before the model is evaluated

    loss_metric = AverageLoss()
    acc_metric = Accuracy(pos_tag_pad_index)
    pred_list: List[int] = []

    for i, (ngrams, pos_tags) in enumerate(test_data.get_batches(100)):
        # Problem 16: Replace the following two lines with your code.
        # The variable batch_loss must contain the loss incurred for the
        # current mini-batch. The variable logits must contain the
        # output of model. The predictions your model makes must be
        # saved to pred_list. Do not edit anything in this function
        # above this line.
        output = model(ngrams) 
        logits = output.reshape(-1, output.shape[2])
        pos_tags = pos_tags.flatten()
        batch_loss = loss_function(logits, pos_tags)
        pred_list += list(logits[pos_tags != pos_tag_pad_index].argmax(dim=-1))

        # Update metrics. Do not edit anything in this function below
        # this line.
        _ = loss_metric.update(batch_loss, len(ngrams))
        _ = acc_metric.update(logits, pos_tags)

    # If you keep a list of the network outputs and the targets in the  
    # variables pred_list and target_list, the following lines of code
    # will create and print a confusion matrix for you.
    if print_conf_matrix:
        print_cm(pred_list, list(itertools.chain(*test_data.labels)), model.pos_tag_vocab.forms[:-1])

    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, message)
    return loss_metric.value, acc_metric.value


def run_trial(model: Union[MLPPosTagger, RNNPosTagger], 
              train_data: Dataset, dev_data: Dataset, test_data: Dataset, 
              pos_tag_pad_index: int,
              num_epochs: int = 10, batch_size: int = 64, lr: float = 1., 
              filename: str = "checkpoint.pt", print_conf_matrix: bool = False) \
        -> Union[MLPPosTagger, RNNPosTagger]:
    """
    Problems 17–21: Use this function to train a model with one
    particular configuration of hyperparameters.

    DATASET HYPERPARAMETERS:

    :param train_data: The training data
    :param dev_data: The validation data
    :param test_data: The testing data
    :param pos_tag_pad_index:

    MODEL HYPERPARAMETERS:

    :param model: The model that will be trained
    :param filename: The filename to save checkpoints to. It should end
        with ".pt"

    TRAINING LOOP HYPERPARAMETERS:

    :param num_epochs: The number of epochs to train the model for
    :param batch_size: The size of the mini-batches used to train the
        model

    ADAM HYPERPARAMETERS:

    :param lr: The initial learning rate used for SGD

    OTHER SETTINGS:

    :param print_conf_matrix: Prints out a confusion matrix if True

    :return: The trained model
    """
    # Create model
    loss_function = nn.CrossEntropyLoss(ignore_index=pos_tag_pad_index,
                                        reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    best_dev_acc = 0.
    best_epoch = None
    for i in range(num_epochs):
        print("Epoch {}".format(i))

        # Train model
        train_epoch(model, train_data, batch_size, loss_function, optimizer,
                    pos_tag_pad_index)
        _, dev_acc = evaluate(model, dev_data, loss_function,
                              "Epoch {} Validation".format(i),
                              pos_tag_pad_index)

        # Save checkpoint
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = i

            with timer("Saving checkpoint..."):
                torch.save(model.state_dict(), filename)

    print("The best validation accuracy of {:.3f} occurred after epoch {}."
          "".format(dev_acc, best_epoch))

    model.load_state_dict(torch.load(filename))
    _ = evaluate(model, test_data, loss_function, "Test", pos_tag_pad_index,
                 print_conf_matrix=print_conf_matrix)

    return model
