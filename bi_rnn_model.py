from keras.models import Sequential
from keras.layers import Embedding, Dense, RNN, Bidirectional
from config import VOCAB_SIZE


def build_bi_rnn_model(max_len, embedding_len=128, rnn_units=64):
    """
    Functionality to build a simple bidirectional rnn model with 64 rnn units.

    :param max_len: maximum input length of the sequences
    :param embedding_len: the dimension of the output of the embedding layer
    :param lstm_units: the number of rnn units required
    :return: the rnn model
    """

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    model.add(Bidirectional(RNN(rnn_units)))
    model.add(Dense(1, activation='sigmoid'))
    return model
