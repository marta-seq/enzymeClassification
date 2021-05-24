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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

# PAD ZEROS 200 20 aa  X = 0 categorical encoding
def pad_sequence(df, seq_len=700, padding='pre', truncating='pre', alphabet = "XARNDCEQGHILKMFPSTWYV"):
    # sequences_original = df['sequence'].tolist()
    # sequences=[]
    # for seq in sequences_original:
    #     seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
    #     seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
    #     seq3 = seq2.replace('U',
    #                         'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
    #     seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
    #     sequences.append(seq4)
    sequences = df

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # {'X': 0,
    #  'A': 1,
    #  'R': 2,
    #  'N': 3,
    #  'D': 4,...
    sequences_integer_ecoded = []
    for seq in sequences:
        # seq = seq.replace('X', 0)  # unknown character eliminated
        # define a mapping of chars to integers
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)
    fps_x = pad_sequences(sequences_integer_ecoded, maxlen=seq_len, padding=padding, truncating=truncating, value=0.0)   # (4042, 200)
    return fps_x


# FROM UNIREF 90 DATASET
# Get dataset
hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
lev_1_single_label = get_ec_1_level(hot_90, single_label=True)
# lev_1_single_label = get_ec_2_level_more_than_x_samples(hot_90, x=50, single_label=True)
# lev_1_single_label = get_ec_3_level_more_than_x_samples(hot_90, x=50, single_label=True)
# lev_1_single_label = get_ec_complete_more_than_x_samples(hot_90, x=50, single_label=True)

lev_1_single_label = lev_1_single_label.loc[lev_1_single_label['sequence'].str.contains('!!!') == False]
lev_1_single_label = remove_zeros(column='ec_single_label', data=lev_1_single_label)  # without zeros
lev_1_single_label = lev_1_single_label.dropna(subset=['sequence'])
print(get_counts(column=lev_1_single_label['ec_single_label']))
seq_len = 500
label = lev_1_single_label['ec_single_label']
fps_y_encoded, fps_y_hot, ecs = binarize_labels(label)
alphabet = "ARNDCEQGHILKMFPSTWYV"
alphabet_x = "XARNDCEQGHILKMFPSTWYV"
alphabet_all_characters = "XARNDCEQGHILKMFPSTWYVBZUO"
sequences = deal_with_strange_aa(sequences=lev_1_single_label['sequence'],
                                 alphabet=alphabet)  # in this case will substitute strange aa and supress X

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
fps_x = pad_sequence(sequences, seq_len=seq_len, padding='post', truncating='pre',
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

x_train, x_test, x_dval, y_train, y_test, y_dval = \
    divide_dataset(fps_x_hot, fps_y_encoded, test_size=0.2, val_size=0.2)
vector_size = x_train.shape[1]
final_units = fps_y_hot.shape[1]

# https://github.com/philipperemy/keract
# https://medium.com/asap-report/visualizing-lstm-networks-part-i-f1d3fa6aace7
# https://www.mathworks.com/help/deeplearning/ug/visualize-features-of-lstm-network.html


tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

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



def bilstm_attention_context(input_dim, number_classes,
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
        # input dim timesteps = seq size , features. 21 features per character
        input = Input(shape=(input_dim, n_features,), dtype='float32', name='main_input')(model)
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        masking = Masking(mask_value=0)(input)
        lstm = Bidirectional(
            LSTM(units=lstm_layers[0], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[0], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(masking)
        lstm2 = Bidirectional(
            LSTM(units=lstm_layers[1], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[1], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(lstm)
        lstm3 = Bidirectional(
            LSTM(units=lstm_layers[2], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[2], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(lstm2)

        a, context = attention()(lstm3)
        # add denses
        model = Dense(units=dense_layers[0], activation=dense_activation,
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(a)
        model = BatchNormalization()(model)
        model = Dropout(dropout_rate_dense[0])(model)
        model = Dense(units=dense_layers[1], activation=dense_activation,
                      kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(model)
        model = BatchNormalization()(model)
        model = Dropout(dropout_rate_dense[1])(model)

        # Add Classification Dense, Compile model and make it ready for optimization
        model= (Dense(number_classes, activation='softmax'))(model)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units,
                        n_features=21)

fps_x_hot = fps_x_hot.astype(np.int8)

dl_path = '/home/amsequeira/enzymeClassification/models/try_attent'
report_name = str(dl_path +
                  'try_attentions_context')
model_name = str(dl_path)

dl = DeepML(x_train=x_train, y_train=y_train, x_test=x_test.astype(np.int8),
            y_test=y_test,
            number_classes=final_units, problem_type='multiclass',
            x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=5, batch_size=64,
            path=dl_path,
            report_name=report_name, validation_split=0.2,
            reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
            early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
            verbose=1)
model = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()
print(scores)
print(report)
print(cm)
dl.save_model(model_name)
model[0].model.save(model_name)

model = dl.get_model()
model.model.save(model_name)

model_reuse = load_model(model_name)
model.summary()
model_reuse.summary()
########################################################################################################################
# try keract
# https://github.com/philipperemy/keract
import keract
activations = keract.get_activations(model_reuse, x_train[:1], layer_names=None, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
# activations = keract.get_activations(model_reuse, x_train[:1],
# layer_names=['bidirectional_12', 'bidirectional_13', 'bidirectional_14','attention_4', 'dense_12', 'dense_13', 'dense_14'],
# nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
keract.display_activations(activations, cmap=None, save=False, directory='.', data_format='channels_last', fig_size=(24, 24), reshape_1d_layers=False)

########################################################################################################################
# https://github.com/mjDelta/attention-mechanism-keras/blob/master/attention_lstm.py
from matplotlib import pyplot as plt
TIME_STEPS=500
def get_activation(model,layer_name,inputs):
    layer=[l for l in model.layers if l.name==layer_name][0]

    func=K.function([model.input],[layer.output])

    return func([inputs])[0]

act = get_activation(model_reuse,layer_name='attention_4',inputs=fps_x_hot[:50])
attention_probs=np.mean(get_activation(model_reuse,"attention_4",fps_x_hot[:50]),axis=0)
top_att=sorted(range(len(attention_probs)), key=lambda i: attention_probs[i])[-10:]
top_att = sorted(range(len(act[49])), key=lambda i: attention_probs[i])[-10:]
# todo
# see mean of activations per time step
# put by class
layer_name = 'bidirectional_14'
act = get_activation(model_reuse,layer_name=layer_name,inputs=x_test[:50])
act_probs=np.mean(act,axis=0) # get means of lstm units for each timestep
act_probs2=np.mean(act_probs,axis=1) # get means of sequences for each timestep
plt.plot(act_probs2)
plt.title("LSTM attention probs")
plt.show()

# get weights it does not give nay information. is values from units of layers
# https://stackoverflow.com/questions/57012563/interpreting-get-weight-in-lstm-model-in-keras
def get_weights(model,layer_name):
    layer=[l for l in model.layers if l.name==layer_name][0]
    config = layer.get_config()
    wei = layer.get_weights()
    print(layer.get_config(), layer.get_weights())
    return wei

wei = get_weights(model_reuse,layer_name='bidirectional_14')


# #################################################################################################
# https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff

from keras.utils import np_utils
import re

# Imports for visualisations
from IPython.display import HTML as html_print
from IPython.display import display
import keras.backend as K
from tensorflow.python.keras import backend


# layers
# [<tensorflow.python.keras.layers.core.Masking at 0x7f00fb0bd550>,
# <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f00fb0bd910>,
# <tensorflow.python.keras.layers.wrappers.Bidirectional at 0x7f00fb0bddc0>,
# <tensorflow.python.keras.layers.wrappers.Bidirectional at 0x7f00fafebf40>,
# <tensorflow.python.keras.layers.wrappers.Bidirectional at 0x7f00faf9cf40>,
# <tensorflow.python.keras.saving.saved_model.load.attention at 0x7f00fafb7c40>,    5
# <tensorflow.python.keras.layers.core.Dense at 0x7f00faf46d30>,
# <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x7f00faf4cc40>,
# <tensorflow.python.keras.layers.core.Dropout at 0x7f00faf5d580>,
# <tensorflow.python.keras.layers.core.Dense at 0x7f00faf60220>,
# <tensorflow.python.keras.layers.normalization_v2.BatchNormalization at 0x7f00faf660a0>,
# <tensorflow.python.keras.layers.core.Dropout at 0x7f00faf6d9a0>,
# <tensorflow.python.keras.layers.core.Dense at 0x7f00faf72640>]
# Backend Function to get Intermediate Layer Output
# visualise outputs of second LSTM layer i.e. third layer in the whole architecture.
# attn_func will return a hidden state vector of size 512. These will be activations of LSTM layer with 512 units.
def get_activation(model,layer_name,inputs):
    layer=[l for l in model.layers if l.name==layer_name][0]

    func=K.function([model.input],[layer.output])
    # dont know difference. the first one is from above. the second from the url
    # func = K.function(inputs = [model.get_input_at(0), backend.symbolic_learning_phase()],
    #                 outputs = [layer.output])
    return func([inputs])[0]


# These helper functions will help us visualise character sequence with each of their activation values. We are
# passing the activations through sigmoid function as we need values in a scale that can denote their importance to the
# whole output. get_clr function helps get appropriate colour for a given value.
# get html element
def cstr(s, color='black'):
    if s == ' ':
        return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
    else:
        return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

# print html
def print_color(t):
    display(html_print(''.join([cstr(ti, color=ci) for ti,ci in t])))

# get appropriate color for value
def get_clr(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
                                                          '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
              '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
              '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value * 100) / 5)
    return colors[value]

# sigmoid function
def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z
# After applying sigmoid on the layer output, the values lie in the range 0 to 1. Closer the number is to 1, higher
# importance it has. If the number is closer to 0, it is meant to not contribute in any major way to the final prediction.
# The importance of these cells is denoted by the colour, where Blue denotes lower importance and Red denotes higher
# importance.


# visualize function takes as input the predicted sequence, the sigmoid values for each character in the sequence and the
# cell number to visualise. Based on the value of the output, character is printed with an appropriate background colour.
def visualize(output_values, seq, cell_no):
    print("\nCell Number:", cell_no, "\n")
    text_colours = []
    for i in range(len(seq)):
        text = (seq[i], get_clr(output_values[i][int(cell_no-1)]))
        text_colours.append(text)
    print_color(text_colours)
    text_colours.show_batch()
    plt.show()

# Get Predictions from random sequence
# get_predictions function randomly chooses an input seed sequence and gets the predicted
# sequence for that seed sequence.

def get_predictions(model_name, layer_name, data):
    # start = np.random.randint(0, len(data)-1)
    # pattern = data[start]
    result_list, output_values = [], []

    # Prediction
    prediction = model_reuse.predict(data, verbose=0)

    # LSTM Activations
    output = get_activation(model = model_name, layer_name=layer_name, inputs = data)[0]
    output = sigmoid(output)
    output_values.append(output)

    # Saving generated characters
    result_list.append(prediction)
    return output, prediction

# More than 90% of the cells do not show any understandable patterns. I visualised all 512 cells manually and noticed
# three of them (189, 435, 463) to show some understandable patterns.
seq_test=fps_x_hot[0]
true_seq = sequences[0]
seq_test = seq_test.reshape(1,seq_test.shape[0], seq_test.shape[1]) # , seq_test.shape[1])
output_values, result_list = get_predictions(model_reuse, 'bidirectional_14', seq_test)

for cell_no in top_att: # list with attention higher
    visualize(output_values,true_seq , cell_no)

#######
# ceck
# https://stackoverflow.com/questions/59017288/how-to-visualize-rnn-lstm-gradients-in-keras-tensorflow
# https://matthewmcateer.me/blog/getting-started-with-attention-for-classification/
########################################################################################################################
# try the one from
# https://github.com/philipperemy/keras-attention-mechanism/blob/master/examples/find_max.py
# they have their own class attention
# package attention

import matplotlib.pyplot as plt
import numpy as np
from keract import get_activations
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, LSTM

from attention import Attention


class VisualizeAttentionMap(Callback):

    def __init__(self, model, x):
        super().__init__()
        self.model = model
        self.x = x

    def on_epoch_begin(self, epoch, logs=None):
        attention_map = get_activations(self.model, self.x, layer_names='attention_weight')['attention_weight']
        x = self.x[..., 0]
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 6))
        maps = [attention_map, create_argmax_mask(attention_map), create_argmax_mask(x)]
        maps_names = ['attention layer', 'attention layer - argmax()', 'ground truth - argmax()']
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(maps[i], interpolation='none', cmap='jet')
            ax.set_ylabel(maps_names[i] + '\n#sample axis')
            ax.set_xlabel('sequence axis')
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        cbar_ax = fig.add_axes([0.75, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(f'Epoch {epoch} - training')
        plt.show()


def create_argmax_mask(x):
    mask = np.zeros_like(x)
    for i, m in enumerate(x.argmax(axis=1)):
        mask[i, m] = 1
    return mask

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
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
        model.add(Attention())
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

def try_attention_1():
    model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units,
                            n_features=21)
    dl_path = '/home/amsequeira/enzymeClassification/models'
    report_name = str(dl_path +
                      '/try_attention_visualize')
    model_name = str(dl_path)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=30)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=50, min_lr=0.00001, verbose=1)

    filepath = os.path.join(dl_path, 'weights-{{epoch:02d}}-{{val_loss:.2f}}.hdf5')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0,
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', period=1)
    visualize = VisualizeAttentionMap(model, pd.DataFrame(fps_x_hot).sample(100)) # todo nao me deixa passar um array 2D. n testei
    callbacks = [es, reduce_lr, cp, visualize]

    dl = DeepML(x_train=fps_x_hot, y_train=fps_y_encoded, x_test=x_test.astype(np.int8),
                y_test=y_test,
                number_classes=final_units, problem_type='multiclass',
                x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
                path=dl_path,
                report_name=report_name, validation_split=0.2,
                callbacks=callbacks,
                reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                verbose=1)

    # visualize the attention on the first samples.
    # visualize = VisualizeAttentionMap(model, pd.DataFrame(fps_x_hot).sample(100))
    # model.fit(x_data, y_data, epochs=max_epoch, validation_split=0.2, callbacks=[visualize])


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


########################################################################################################################
# https://www.kaggle.com/alber8295/bigru-w-attention-visualized-for-beginners/notebook
# visualization of a biGRU with attention next
from keras import initializers, regularizers, constraints

# Visualization
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
sns.set()


def visualize_attention():
    # Make new model for output predictions and attentions
    model_att = Model(inputs=model.input, \
                      outputs=[model.output, model.get_layer('attention_vec').output])
    idx = np.random.randint(low = 0, high=X_te.shape[0]) # Get a random test
    tokenized_sample = np.trim_zeros(X_te[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict(X_te[idx:idx+1]) # Perform the prediction

    # Get decoded text and labels
    id2word = dict(map(reversed, tokenizer.word_index.items()))
    decoded_text = [id2word[word] for word in tokenized_sample]

    # Get classification
    label = np.argmax((label_probs>0.5).astype(int).squeeze()) # Only one
    label2id = ['Sincere', 'Insincere']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    for token, attention_score in zip(decoded_text, attentions[0][-len(tokenized_sample):]):
        token_attention_dic[token] = attention_score


    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                            token)
    #html_text += "</p><br>"
    #html_text += "<p style='font-size: large'><b>Classified as:</b> "
    #html_text += label2id[label]
    #html_text += "</p>"

    # Display text enriched with attention scores
    display(HTML(html_text))

    # PLOT EMOTION SCORES

    _labels = ['sincere', 'insincere']
    plt.figure(figsize=(5,2))
    plt.bar(np.arange(len(_labels)), label_probs.squeeze(), align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
    plt.xticks(np.arange(len(_labels)), _labels)
    plt.ylabel('Scores')
    plt.show()

def under_sample(train_df):
    # UNDER SAMPLE
    insincere = len(train_df[train_df.target == 1])
    insincere_indices = train_df[train_df.target == 1].index

    sincere_indices = train_df[train_df.target == 0].index
    random_indices = np.random.choice(sincere_indices, insincere, replace=False)

    under_sample_indices = np.concatenate([insincere_indices,random_indices])
    under_sample = train_df.loc[under_sample_indices]
    train_df = under_sample.sample(frac=1)
    train_df.info()
    return train_df

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
def visualize_attention():
    # Make new model for output predictions and attentions
    '''
    model.get_layer('attention_vec').output:
    attention_vec (Attention)    [(None, 128), (None, 54)] <- We want (None,54) that is the word att
    '''
    model_att = Model(inputs=model.input, \
                      outputs=[model.output, model.get_layer('attention_vec').output[-1]])
    idx = np.random.randint(low = 0, high=x_test.shape[0]) # Get a random test
    tokenized_sample = np.trim_zeros(x_test[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict(x_test[idx:idx+1]) # Perform the prediction

    # Get decoded text and labels
    id2word = dict(map(reversed, char_to_int.items()))
    decoded_text = [id2word[word] for word in tokenized_sample]

    # Get classification
    label = (label_probs>0.5).astype(int).squeeze() # Only one
    label2id = ['Sincere', 'Insincere']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0

    attentions_text = attentions[0,-len(tokenized_sample):]
    #plt.bar(np.arange(0,len(attentions.squeeze())), attentions.squeeze())
    #plt.show();
    #print(attentions_text)
    attentions_text = (attentions_text - np.min(attentions_text)) / (np.max(attentions_text) - np.min(attentions_text))
    for token, attention_score in zip(decoded_text, attentions_text):
        #print(token, attention_score)
        token_attention_dic[token] = attention_score


    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                            token)
    #html_text += "</p><br>"
    #html_text += "<p style='font-size: large'><b>Classified as:</b> "
    #html_text += label2id[label]
    #html_text += "</p>"

    # Display text enriched with attention scores
    display(HTML(html_text))

    # PLOT EMOTION SCORES
    _labels = ['sincere', 'insincere']
    probs = np.zeros(2)
    probs[1] = label_probs
    probs[0] = 1- label_probs
    plt.figure(figsize=(5,2))
    plt.bar(np.arange(len(_labels)), probs.squeeze(), align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
    plt.xticks(np.arange(len(_labels)), _labels)
    plt.ylabel('Scores')
    plt.show()

# Util classes
class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_attention=False,
                 **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
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
        embedded_inputs = (Input(shape=(input_dim, n_features,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        embedded_inputs = Masking(mask_value=0)(embedded_inputs)
        rnn_outs_1 = Bidirectional(
            LSTM(units=lstm_layers[0], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[0], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(embedded_inputs)
        rnn_outs_2 = Bidirectional(
            LSTM(units=lstm_layers[1], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[1], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(rnn_outs_1)
        rnn_outs = Bidirectional(
            LSTM(units=lstm_layers[2], return_sequences=True, activation=activation,
                 recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[2], recurrent_dropout=0.0), input_shape=(input_dim, 20,))(rnn_outs_2)

        # Attention Mechanism - Generate attention vectors
        sentence, word_scores = Attention(return_attention=True, name = "attention_vec")(rnn_outs)

        for layer in range(len(dense_layers)):
            fc = Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(fc)
            fc = BatchNormalization()(fc)
            fc = Dropout(dropout_rate_dense[layer])(fc)
        output = Dense(number_classes, activation='softmax')(fc)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

def try_attention_2():
    model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units,
                            n_features=21)
    dl_path = '/home/amsequeira/enzymeClassification/models'
    report_name = str(dl_path +
                      '/try_attention_visualize')
    model_name = str(dl_path)

    dl = DeepML(x_train=fps_x_hot, y_train=fps_y_encoded, x_test=x_test.astype(np.int8),
                y_test=y_test,
                number_classes=final_units, problem_type='multiclass',
                x_dval=x_dval.astype(np.int8), y_dval=y_dval, epochs=100, batch_size=64,
                path=dl_path,
                report_name=report_name, validation_split=0.2,
                reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
                early_stopping_patience=30, reduce_lr_patience=20, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
                verbose=1)

    model = dl.run_model(model) # todo dá me erro add_weight() got multiple values for argument 'name'
    # https://github.com/keras-team/keras/issues/13540 presents solution but dont think it applies
    scores, report, cm, cm2 = dl.model_complete_evaluate()
    print(scores)
    print(report)
    print(cm)
    dl.save_model(model_name)
    model = dl.get_model()
    model.model.save(model_name)

    model = load_model(model_name)
    model.summary()


    # Credit assignment is allocating importance to input features through the weights of the neural network’s model.
    # This is exactly what an attention layer does. The attention layer allocates more or less importance to each part of
    # the input, and it learns to do this while training. In this visualization, the input sentences are displayed.
    # The background colour varies between white and red. More intense red means more attention given to the word.
    for _ in range(3):
        visualize_attention()


