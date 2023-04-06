from train import run_trial
from data_loader import load_embeddings, all_pos_tags, Dataset, Vocabulary

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

    # Train with GloVe embeddings
    run_trial(train_data, dev_data, test_data, token_vocab, pos_tag_vocab,
              glove_embeddings, "checkpoint1.p", lr=.1, num_epochs=4,
              num_layers=1, step_size=4, gamma=.25, batch_size=5,
              hidden_size=30)
    
    # Train with randomly initialized embeddings
    run_trial(train_data, dev_data, test_data, token_vocab, pos_tag_vocab,
              None, "checkpoint2.p", lr=.1, num_epochs=4,
              num_layers=1, step_size=4, gamma=.25, batch_size=5,
              hidden_size=30)
