import sys
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import operator
import random
from random import sample
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# logging.disable(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf
from itertools import chain
from sklearn.feature_selection import mutual_info_classif

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from collections import Counter

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import re
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import StratifiedKFold
# from nlf_blosum_encoding import blosum_encode
from tensorflow import keras
from tensorflow.keras.layers import LSTM, ConvLSTM2D, BatchNormalization
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, classification_report, \
    multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import ShuffleSplit, train_test_split

from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Convolution2D, GRU, TimeDistributed, Reshape, \
    MaxPooling2D, Convolution1D, BatchNormalization, Masking

# input is 21 categorical 200 paded aa sequences
from keras.layers.merge import concatenate

from get_ec import get_ec_1_level, binarize_labels, get_ec_2_level_more_than_x_samples, remove_zeros, \
    get_ec_3_level_more_than_x_samples, \
    get_ec_complete_more_than_x_samples
from dataset_characterization import get_counts
from get_prot_representation import pad_sequence, deal_with_strange_aa
from deep_ml import DeepML

random.seed(777)


def divide_dataset(fps_x, fps_y, test_size=0.2, val_size=0.1):
    # divide in train, test and validation
    x_train_1, x_test, y_train_1, y_test = train_test_split(fps_x, fps_y, test_size=test_size, random_state=42,
                                                            shuffle=True, stratify=fps_y)

    # iterative_train_test_split(fps_x, fps_y, test_size=test_size)
    train_percentage = 1 - test_size
    val_size = val_size / train_percentage

    x_train, x_dval, y_train, y_dval = train_test_split(x_train_1, y_train_1, test_size=val_size, random_state=42,
                                                        shuffle=True, stratify=y_train_1)

    # stratify=y_train_1, shuffle=True)

    return x_train, x_test, x_dval, y_train, y_test, y_dval


#
#
# FROM UNIREF 90 DATASET
# Get dataset
hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
# lev_1_single_label = get_ec_1_level(hot_90, single_label=True)
# lev_1_single_label = get_ec_2_level_more_than_x_samples(hot_90, x=50, single_label=True)
# lev_1_single_label = get_ec_3_level_more_than_x_samples(hot_90, x=50, single_label=True)
lev_1_single_label = get_ec_complete_more_than_x_samples(hot_90, x=50, single_label=True)

lev_1_single_label = lev_1_single_label.loc[lev_1_single_label['sequence'].str.contains('!!!') == False]
lev_1_single_label = remove_zeros(column='ec_single_label', data=lev_1_single_label)  # without zeros
lev_1_single_label = lev_1_single_label.dropna(subset=['sequence'])
print(get_counts(column=lev_1_single_label['ec_single_label']))
# counts_single_label = get_counts(column=lev_1_single_label['ec_number1'])
# [('2', 54343),
#  ('3', 34484),
#  ('0', 22708),
#  ('1', 19066),
#  ('6', 16290),
#  ('4', 12924),
#  ('5', 8081),
#  ('7', 6860)]
seq_len = 500
# print(lev_1_single_label.shape)
# lev_1_single_label = lev_1_single_label[lev_1_single_label['sequence'].str.len() < seq_len]
# print(lev_1_single_label.shape)
# print(get_counts(column=lev_1_single_label['ec_number1']))
label = lev_1_single_label['ec_single_label']
fps_y_encoded, fps_y_hot, ecs = binarize_labels(label)
alphabet = "ARNDCEQGHILKMFPSTWYV"
alphabet_x = "XARNDCEQGHILKMFPSTWYV"
alphabet_all_characters = "XARNDCEQGHILKMFPSTWYVBZUO"
sequences = deal_with_strange_aa(sequences=lev_1_single_label['sequence'],
                                 alphabet=alphabet)  # in this case will substitute strange aa and supress X
#
# from truncating import get_middle, get_terminals
# seq_new_list = []
# for seq in sequences:
#     # seq_new = get_middle(seq,seq_len)
#     seq_new = get_terminals(seq,seq_len)
#     seq_new_list.append(seq_new)
# print(len(max(seq_new_list, key=len)))
# sequences = seq_new_list
# print(len(max(sequences, key=len)))
print(tf.executing_eagerly())
print('execution eager')
fps_x, remake_sequence = pad_sequence(sequences, seq_len=seq_len, padding='post', truncating='post',
                                      alphabet="XARNDCEQGHILKMFPSTWYV")
# remake sequence is the sequence padded with haracters of aa. the alphabet needs to have X. for the A not be the Zero and not be padded.

print(fps_x)
print(fps_x.shape)
fps_x_hot = to_categorical(fps_x)
# # (174756, 1500, 21)
print(fps_x_hot.shape)
# # print(fps_y_encoded.shape)
# # print(label.shape)
# print(lev_1_single_label.shape)
fps_x_hot_flat = fps_x_hot.reshape(fps_x_hot.shape[0], fps_x_hot.shape[1] * fps_x_hot.shape[2])

# ## Z scales
# zscale = {
#     'A': [0.24,  -2.32,  0.60, -0.14,  1.30], # A
#     'C': [0.84,  -1.67,  3.71,  0.18, -2.65], # C
#     'D': [3.98,   0.93,  1.93, -2.46,  0.75], # D
#     'E': [3.11,   0.26, -0.11, -0.34, -0.25], # E
#     'F': [-4.22,  1.94,  1.06,  0.54, -0.62], # F
#     'G': [2.05,  -4.06,  0.36, -0.82, -0.38], # G
#     'H': [2.47,   1.95,  0.26,  3.90,  0.09], # H
#     'I': [-3.89, -1.73, -1.71, -0.84,  0.26], # I
#     'K': [2.29,   0.89, -2.49,  1.49,  0.31], # K
#     'L': [-4.28, -1.30, -1.49, -0.72,  0.84], # L
#     'M': [-2.85, -0.22,  0.47,  1.94, -0.98], # M
#     'N': [3.05,   1.62,  1.04, -1.15,  1.61], # N
#     'P': [-1.66,  0.27,  1.84,  0.70,  2.00], # P
#     'Q': [1.75,   0.50, -1.44, -1.34,  0.66], # Q
#     'R': [3.52,   2.50, -3.50,  1.99, -0.17], # R
#     'S': [2.39,  -1.07,  1.15, -1.39,  0.67], # S
#     'T': [0.75,  -2.18, -1.12, -1.46, -0.40], # T
#     'V': [-2.59, -2.64, -1.54, -0.85, -0.02], # V
#     'W': [-4.36,  3.94,  0.59,  3.44, -1.59], # W
#     'Y': [-2.54,  2.44,  0.43,  0.04, -1.47], # Y
#     '-': [0.00,   0.00,  0.00,  0.00,  0.00], # -
# }
# fps_x, remake_sequence = pad_sequence(sequences, seq_len=seq_len, padding='post', truncating='pre', alphabet="-ARNDCEQGHILKMFPSTWYV")
# # the alphabet needs to be with -. to - be the 0 and then when recoverting after the padding be the 0. and not get the A
#
#
# #remake sequences does not have zeros
# new_seqs = []
# for sequence in remake_sequence:
#     new_sequence=[zscale[aa] for aa in sequence]
#     new_seqs.append(new_sequence)
# fps_x = np.array(new_seqs)
# print(fps_x)
# print(fps_x.shape)


# # BLOSUM
# blosum62 = {
#     'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
#     'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
#     'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
#     'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
#     'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
#     'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
#     'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
#     'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
#     'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
#     'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
#     'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
#     'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
#     'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
#     'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
#     'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
#     'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
#     'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
#     'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
#     'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
#     'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
#     '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
# }
#
# fps_x, remake_sequence = pad_sequence(sequences, seq_len=seq_len, padding='post', truncating='pre',
#                                       alphabet="-ARNDCEQGHILKMFPSTWYV")
# # the alphabet needs to be with -. to - be the 0 and then when recoverting after the padding be the 0. and not get the A
#
#
# # remake sequences does not have zeros. has -
# new_seqs = []
# for sequence in remake_sequence:
#     new_sequence = [blosum62[aa] for aa in sequence]
#     new_seqs.append(new_sequence)
# fps_x = np.array(new_seqs)
# print(fps_x)
# print(fps_x.shape)


dl_path = '/home/amsequeira/enzymeClassification/models'
report_name = str(dl_path +
                  '/attention_to_viz')
model_name = str(dl_path)

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


def bilstm_simple(input_dim, number_classes,
                  optimizer='Adam',
                  lstm_layers=(64, 64, 32),
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  dropout_rate=(0.1, 0.1, 0.1),
                  l1=1e-5, l2=1e-4,
                  dense_layers=(32, 16),
                  dropout_rate_dense=(0.1, 0.1),
                  dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers) - 1):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)

        # add last lstm layer
        model.add(Bidirectional(
            LSTM(units=lstm_layers[-1], return_sequences=False,
                 activation=activation, recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[-1], recurrent_dropout=0.0)))

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model


def lstm_simple(input_dim, number_classes,
                optimizer='Adam',
                lstm_layers=(64, 64, 32),
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout_rate=(0.1, 0.1, 0.1),
                l1=1e-5, l2=1e-4,
                dense_layers=(32, 16),
                dropout_rate_dense=(0.1, 0.1),
                dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers) - 1):
            model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                           recurrent_activation=recurrent_activation,
                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                           dropout=dropout_rate[layer], recurrent_dropout=0.0))

        # add last lstm layer
        model.add(
            LSTM(units=lstm_layers[-1], return_sequences=False,
                 activation=activation, recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[-1], recurrent_dropout=0.0))

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model


def gru_simple(input_dim, number_classes,
               optimizer='Adam',
               lstm_layers=(64, 64, 32),
               activation='tanh',
               recurrent_activation='sigmoid',
               dropout_rate=(0.1, 0.1, 0.1),
               l1=1e-5, l2=1e-4,
               dense_layers=(32, 16),
               dropout_rate_dense=(0.1, 0.1),
               dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers) - 1):
            model.add(GRU(units=lstm_layers[layer], return_sequences=True, activation=activation,
                          recurrent_activation=recurrent_activation,
                          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                          dropout=dropout_rate[layer], recurrent_dropout=0.0))

        # add last lstm layer
        model.add(
            GRU(units=lstm_layers[-1], return_sequences=False,
                activation=activation, recurrent_activation=recurrent_activation,
                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                dropout=dropout_rate[-1], recurrent_dropout=0.0))

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


def bilstm_attention(input_dim, number_classes,
                     n_features=20,
                     optimizer='Adam',
                     lstm_layers=(64, 64, 32),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout_rate=(0.1, 0.1, 0.1),
                     l1=1e-5, l2=1e-4,
                     dense_layers=(32, 16),
                     dropout_rate_dense=(0.1, 0.1),
                     dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, n_features,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)

        # receives LSTM with return sequences =True

        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model


def lstm_attention(input_dim, number_classes,
                   optimizer='Adam',
                   lstm_layers=(64, 64, 32),
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   dropout_rate=(0.1, 0.1, 0.1),
                   l1=1e-5, l2=1e-4,
                   dense_layers=(32, 16),
                   dropout_rate_dense=(0.1, 0.1),
                   dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 20,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers)):
            model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                           recurrent_activation=recurrent_activation,
                           kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                           dropout=dropout_rate[layer], recurrent_dropout=0.0))

        # receives LSTM with return sequences =True

        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        return model


def gru_attention(input_dim, number_classes,
                  optimizer='Adam',
                  lstm_layers=(64, 64, 32),
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  dropout_rate=(0.1, 0.1, 0.1),
                  l1=1e-5, l2=1e-4,
                  dense_layers=(32, 16),
                  dropout_rate_dense=(0.1, 0.1),
                  dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 21,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(GRU(units=lstm_layers[layer], return_sequences=True, activation=activation,
                          recurrent_activation=recurrent_activation,
                          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                          dropout=dropout_rate[layer], recurrent_dropout=0.0)))

        # receives LSTM with return sequences =True

        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())

        return model


def bilstm_attention_embedding(input_dim, number_classes, output_dim,
                               optimizer='Adam',
                               lstm_layers=(64, 64, 32),
                               activation='tanh',
                               recurrent_activation='sigmoid',
                               dropout_rate=(0.1, 0.1, 0.1),
                               l1=1e-5, l2=1e-4,
                               dense_layers=(32, 16),
                               dropout_rate_dense=(0.1, 0.1),
                               dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim,), dtype='float32', name='main_input'))
        # add initial dropout
        model.add(Masking(mask_value=0))
        model.add(Embedding(input_dim=len(alphabet) + 1, output_dim=output_dim))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)

        # receives LSTM with return sequences =True

        # add attention
        # model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model


x_train, x_test, x_dval, y_train, y_test, y_dval = \
    divide_dataset(fps_x_hot, fps_y_encoded, test_size=0.2, val_size=0.2)

vector_size = x_train.shape[1]
final_units = fps_y_hot.shape[1]
# model = KerasClassifier(build_fn=bilstm_simple, input_dim=vector_size, number_classes=final_units)
# model = KerasClassifier(build_fn=lstm_simple, input_dim=vector_size, number_classes=final_units)
# model = KerasClassifier(build_fn=gru_simple, input_dim=vector_size, number_classes=final_units)
model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units,
                        n_features=21)
# model = KerasClassifier(build_fn=lstm_attention, input_dim=vector_size, number_classes=final_units)
# model = KerasClassifier(build_fn=gru_attention, input_dim=vector_size, number_classes=final_units)
# model = KerasClassifier(build_fn=bilstm_attention_embedding, input_dim=vector_size, number_classes=final_units,
#                         output_dim=5)

fps_x_hot = fps_x_hot.astype(np.int8)

dl = DeepML(x_train=x_train, y_train=y_train, x_test=x_test.astype(np.int8),
            y_test=y_test,
            number_classes=final_units, problem_type='multiclass',
            x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
            path=dl_path,
            report_name=report_name, validation_split=0.2,
            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
            early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
            verbose=1)

print(fps_x_hot.shape)
test = fps_x_hot.shape[0] / 5
train = fps_x_hot.shape[0] - test
val = train * 0.3
train = train - val
print(train)
print(test)
print(val)


model = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()
print(scores)
print(report)
print(cm)
dl.save_model(model_name)
model = dl.get_model()
model.model.save(model_name)

model = load_model(model_name)
model.summary()

# scores = dl.train_model_cv(x_cv=fps_x_hot.astype(np.float32), y_cv=fps_y_encoded.astype(np.float32), cv=5, model=model)

# print(fps_x)
# dl = DeepML(x_train=fps_x, y_train=fps_y_encoded, x_test=x_test.astype(np.int8),
#             y_test=y_test,
#             number_classes=final_units, problem_type='multiclass',
#             x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
#             path=dl_path,
#             report_name=report_name, validation_split=0.2,
#             reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#             early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#             verbose=1)
# scores = dl.train_model_cv(x_cv=fps_x.astype(np.float32), y_cv=fps_y_encoded.astype(np.float32), cv=5, model=model)

print(scores)

K.clear_session()
tf.keras.backend.clear_session()

#
# x_train, x_test, x_dval, y_train, y_test, y_dval = \
#     divide_dataset(fps_x_hot, fps_y_encoded, test_size=0.2, val_size=0.2)
#
# vector_size = x_train.shape[1]
# final_units = fps_y_hot.shape[1]
# model = KerasClassifier(build_fn=bilstm_simple, input_dim=vector_size, number_classes=final_units)
#
#
#
#
# dl=DeepML(x_train = fps_x_hot.astype(np.float32), y_train = fps_y_encoded,x_test=x_test.astype(np.float32),
#           y_test= y_test,
#           number_classes=final_units, problem_type='multiclass',
#           x_dval=x_dval.astype(np.float32), y_dval=y_dval, epochs=100, batch_size=64,
#           path='/home/amsequeira/enzymeClassification/models/cv_len_pad',
#           report_name=report_name, validation_split=0.2,
#           reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#           early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#           verbose=1)
#
# print(fps_x_hot.shape)
# test = fps_x_hot.shape[0]/5
# train = fps_x_hot.shape[0]-test
# val = train*0.3
# train = train-val
# print(train)
# print(test)
# print(val)
# scores = dl.train_model_cv(x_cv=fps_x_hot.astype(np.float32), y_cv=fps_y_encoded.astype(np.float32), cv=5, model=model)
# print(scores)
#

# dl=DeepML(x_train = x_train.astype(np.float32), y_train = y_train,x_test=x_test.astype(np.float32),
#                     y_test= y_test,
#                     number_classes=final_units, problem_type='multiclass',
#                     x_dval=x_dval.astype(np.float32), y_dval=y_dval, epochs=100, batch_size=64,
#                     path='/home/amsequeira/enzymeClassification/models/',
#           report_name='/home/amsequeira/enzymeClassification/models/128_64_32_2v_lstm', validation_split=0.2,
#                     reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
#                     early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
#                     verbose=1)
#
#
# dense = dl.run_model(model)
# scores, report, cm, cm2 = dl.model_complete_evaluate()
# print(scores)
# print(report)
# print(cm)
# dl.save_model('model.h5')

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer_sgd = keras.optimizers.SGD(learning_rate=lr_schedule)
#
#
# param_grid={'lstm_layers':[(128,64),(64,32),(256,128),(512,256), (512,128), (128,32)],
#             'dropout_rate':[(0.3,0.2), (0.0,0.2), (0.3,0.3), (0.2,0.2), (0.0,0.0), (0.3,0.0)],
#             'dense_layers':[(32,16), (64,32), (64,16), (128,64)],
#             'optimizer':['Adam', 'RMSprop']}
# param_grid2={'lstm_layers':[(128,),(64,),(32,),(512,), (256,), (16,)],
#             'dropout_rate':[(0.3,), (0.0,), (0.2,), (0.1,)],
#             'dense_layers':[(32,16), (64,32), (64,16), (128,64), (32,), (64,),(128,), (16,)],
#             'optimizer':['Adam', 'RMSprop']}
# param_grid3={'lstm_layers':[(128,64,32),(64,32,16),(256,128,64),(512,256,128), (512,128,32), (128,32,16)],
#             'dropout_rate':[(0.3,0.2,0.1), (0.0,0.2,0.2), (0.3,0.3,0.2), (0.2,0.2,0.2), (0.0,0.0,0.0), (0.3,0.0,0.3)],
#             'dense_layers':[(32,16), (64,32), (64,16), (128,64)],
#             'optimizer':['Adam', 'RMSprop']}

# model_opt = dl.get_opt_params(param_grid3,  model, optType='randomizedSearch', cv=5, dataX=None, datay=None,
#                   n_iter_search=15, n_jobs=1, scoring=make_scorer(matthews_corrcoef))
# dl.save_model('model.h5')

# saber a melhor epoch
# best_n_epochs = np.argmin(history.history['val_loss']) + 1

# fazer pip install no github projecto conjunto
# pip install mypypipackage --no-deps --index-url https://:@gitlab.example.com/api/v4/projects//packages/pypi/simple
