
# netflix_ae.py - Fang Xi Lin, 2021

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import argparse

def load_data(data_path = 'data.filtered.csv', shuffle = True):
    """
    This functon loads ChEMBL data, extract the SMILES and AlogP columns,
    then split them into a training (80 %) and test (20%) set.

    Inputs:
        data_path: str, relative path to data file.
        shuffle: bool, whether to shuffle the data before splitting.

    Returns:
        tuples of arrays `(x_train, x_test), (y_train, y_test),
        (sentence_length, num_words, encoding)`

        **x_train, x_test**: str array of chemicals in SMILES format
            with shape (num_data*0.8).

        **y_train, y_test**: float32 array of AlogP values
            with shape (num_data*0.2).

        **sentence_length**: int32 max sentence length.

        **num_words**: int32 vocabulary size.

        **encoding**: dict word to int encoder.
    """

    # Extract the two relevant columns
    df = pd.read_csv(data_path, header=0, usecols=(6,8))

    # Shuffle input data
    if shuffle:
        print('Shuffling data')
        df = df.sample(frac=1)

    smiles = df['Smiles']
    alogp = df['AlogP']

    N = smiles.size

    # Split into test and training, and reindex them from 0
    x_train = smiles[N//5:].reset_index(drop=True)
    y_train = alogp[N//5:].reset_index(drop=True)

    x_test = smiles[:N//5].reset_index(drop=True)
    y_test = alogp[:N//5].reset_index(drop=True)

    # maximum sentence length for zero padding
    smiles_len = [len(word) for word in smiles]
    sentence_length = np.max(smiles_len)
    print('Maximum sentence length is:', sentence_length)

    # number of unique words
    all_strings = ''.join(word for word in smiles)
    words = np.sort(list(set(all_strings)))
    num_words = words.size
    print('We have', num_words, 'different words.')

    # encode the result, add one to allow zero padding character
    encoding = {w: i+1 for i, w in enumerate(words)}
    decoding = {i+1: w for i, w in enumerate(words)}

    return (x_train, y_train), (x_test, y_test), (sentence_length, num_words, encoding)

def preprocessor(data_arr, encoding, max_length):
    """
    Data preprocessor. Takes in sentence array and a word to int encoder,
     and returns an int-encoded array of the sentences,
     where each sentence is zero-padded to max_length.
    """
    data_encoded = []
    for i in range(len(data_arr)):
        sentence_train = data_arr[i]
        data_encoded.append([encoding[word] for word in sentence_train])

    data_padded = pad_sequences(data_encoded, maxlen=max_length, padding='post')
    return data_padded

def get_model(vocab_size, sentence_length, embed_dim = 32, lstm_dim = 256):
    """
    LSTM model. Consists of an embedding layer of input words followed by an
    LSTM layer, and then a linear output layer.

    Embedding layers needs vocab_size + 1 as input because of zero padding.
    mask_zero = True then masks the zero padding for the following layers.
    """

    model = km.Sequential()
    # Embedding is extremely more efficient here than one-hot encoding, both for
    # memory and training. One-hot encoded network almost did not train at all.
    # this makes sense since the actual chemicals must be important for
    # the AlogP value, and one-hot encoding does not take that into account
    # at all.
    model.add(kl.Embedding(vocab_size+1, embed_dim,
        input_length=sentence_length, mask_zero = True))
    model.add(kl.LSTM(lstm_dim))
    model.add(kl.Dense(1, activation = 'linear'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam',
                    metrics = ['mean_squared_error'])

    return model

def main(data_path = 'data.filtered.csv', batch_size = 256, epochs=20):
    print('Loading ChEMBL data')
    (x_train, y_train), (x_test, y_test),  (sentence_length, num_words, encoding) = load_data(data_path)
    print('Preprocessing data for embedding layer')
    x_train_padded = preprocessor(x_train, encoding, sentence_length)
    x_test_padded = preprocessor(x_test, encoding, sentence_length)

    print('Building network')
    model = get_model(vocab_size = num_words, sentence_length=sentence_length, embed_dim = 32, lstm_dim = 256)

    print('Training network')
    try:
        fit = model.fit(x_train_padded, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)
    except:
        pass

    print('Evalutating network')
    train_score = model.evaluate(x_train_padded, y_train, batch_size = batch_size)
    test_score = model.evaluate(x_test_padded, y_test, batch_size = batch_size)

    print('Training score is {}'.format(train_score))
    print('Test score is {}'.format(test_score))
    print('Score difference is {} %'.format(100 * (test_score[0] - train_score[0]) ) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM network for predicting\
                    AlogP values from SMILES')
    parser.add_argument('--data-path', help='Relative path to Netflix\
                    data file', default='data.filtered.csv')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch size',
                    default=256)
    # early stopping at 50 seems to be the best testing score
    parser.add_argument('-e', '--epochs', type=int, help='Number to\
                    epochs to train', default=20)
    args = parser.parse_args()

    main(data_path = args.data_path, batch_size = args.batch_size,
        epochs = args.epochs)
