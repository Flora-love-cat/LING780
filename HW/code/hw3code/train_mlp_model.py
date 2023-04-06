"""
Sample script for training an MLP model. Please feel free to use this
script for Problem 17. You may modify this script as much as you would
like. DO NOT TURN IN THIS FILE WHEN YOU SUBMIT YOUR WORK.
"""
from _utils import timer
from data_loader import load_embeddings, all_pos_tags, Dataset, Vocabulary
from models import MLPPosTagger
from train import run_trial

if __name__ == "__main__":
    # Load embeddings and vocab
    all_tokens, pretrained_embeddings = load_embeddings(
        cache_filename="data/glove.p")

    # Use the 100,000 most common words
    vocab_size = 100000
    all_tokens = all_tokens[:vocab_size]
    pretrained_embeddings = pretrained_embeddings[:vocab_size]

    # Create vocab objects
    token_vocab = Vocabulary(all_tokens + ["[UNK]", "[BOS]", "[PAD]"])
    pos_tag_vocab = Vocabulary(all_pos_tags + ["[PAD]"])
    vocab_size, pretrained_embedding_size = pretrained_embeddings.shape

    # Load datasets
    with timer("Loading data..."):
        train_data = Dataset.from_conll(
            "data/en_ewt-ud-train.conllu", token_vocab, pos_tag_vocab,
            cache_filename="data/train{}.p".format(vocab_size))

        dev_data = Dataset.from_conll(
            "data/en_ewt-ud-dev.conllu", token_vocab, pos_tag_vocab,
            cache_filename="data/dev{}.p".format(vocab_size))

        test_data = Dataset.from_conll(
            "data/en_ewt-ud-test.conllu", token_vocab, pos_tag_vocab,
            cache_filename="data/test{}.p".format(vocab_size))

    # Create a model
    use_pretrained_embeddings = False
    if not use_pretrained_embeddings:
        pretrained_embeddings = None
        embedding_size = 20
    else:
        embedding_size = pretrained_embedding_size


    ngram_length = 3
    embedding_size=20
    hidden_size=30
    num_hidden_layers=2
    num_epochs=30
    batch_size=5 
    lr=.01

    model = MLPPosTagger(
            token_vocab, pos_tag_vocab, ngram_length, 
            embedding_size, hidden_size, num_hidden_layers, pretrained_embeddings) 

    # Train
    run_trial(model, train_data, dev_data, test_data,
                pos_tag_vocab.get_index("[PAD]"),
                num_epochs, batch_size, lr, 
                filename="checkpoint.pt", print_conf_matrix=True)