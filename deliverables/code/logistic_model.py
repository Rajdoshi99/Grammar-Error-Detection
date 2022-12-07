from keras.models import Sequential
from keras.layers import Embedding, Dense

from config import VOCAB_SIZE, MAX_LEN


def build_logistic_model(max_len=MAX_LEN, embedding_len=128):
    """
    Functionality to build a simple logistic model

    :param max_len: maximum input length of the sequences
    :param embedding_len: the dimension of the output of the embedding layer
    :return: the logistic model
    """

    model = Sequential(name='logistic_model')
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    model.add(Dense(1, activation='sigmoid'))
    return model
