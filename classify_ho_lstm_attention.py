import sys
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import operator
import random
from random import sample

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

os.environ["CUDA_VISIBLE_DEVICES"] = '3,2'
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

from get_ec import get_ec_1_level, binarize_labels
from dataset_characterization import get_counts
from get_prot_representation import pad_sequence, deal_with_strange_aa
from deep_ml import DeepML

random.seed(999)


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


# Get dataset
hot_90 = pd.read_csv('/home/amsequeira/deepbio/datasets/ecpred/ecpred_uniprot_uniref_90.csv', low_memory=False)
lev_1_single_label = get_ec_1_level(hot_90, single_label=True)
lev_1_single_label = lev_1_single_label.loc[lev_1_single_label['sequence'].str.contains('!!!') == False]
lev_1_single_label = lev_1_single_label.dropna(subset=['sequence'])

counts_single_label = get_counts(column=lev_1_single_label['ec_number1'])
# [('2', 54343),
#  ('3', 34484),
#  ('0', 22708),
#  ('1', 19066),
#  ('6', 16290),
#  ('4', 12924),
#  ('5', 8081),
#  ('7', 6860)]

label = lev_1_single_label['ec_number1']
fps_y_encoded, fps_y_hot, ecs = binarize_labels(label)
alphabet = "ARNDCEQGHILKMFPSTWYV"
alphabet_x = "XARNDCEQGHILKMFPSTWYV"
alphabet_all_characters = "ARNDCEQGHILKMFPSTWYVXBZUO"
seq_len = 500
sequences = deal_with_strange_aa(sequences=lev_1_single_label['sequence'],
                                 alphabet=alphabet)  # in this case will substitute strange aa and mantain X
fps_x = pad_sequence(sequences, seq_len=seq_len, padding='pre', truncating='pre', alphabet="ARNDCEQGHILKMFPSTWYV")
fps_x_hot = to_categorical(fps_x)
# (174756, 1500, 21)
fps_x_hot_flat = fps_x_hot.reshape(fps_y_hot.shape[0], fps_x_hot.shape[1]*fps_x_hot.shape[2])

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()


# attention
# This can be a possible custom solution with a custom layer which compute attention on the positional/temporal dimension
# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

class Attention(Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(Attention,self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
# define a class named Attention as a derived class of the Layer class. We need to define four functions as per the Keras
# custom layer generation rule. These are build(),call (), compute_output_shape() and get_config().
# Inside build (), we will define our weights and biases, i.e., Wa and B as discussed previously. If the previous LSTM
# layer’s output shape is (None, 32, 100) then our output weight should be (100, 1) and bias should be (100, 1)
# dimensional.
# Inside call (), we will write the main logic of Attention. We simply must create a Multi-Layer Perceptron (MLP).
# Therefore, we will take the dot product of weights and inputs followed by the addition of bias terms. After that,
# we apply a ‘tanh’ followed by a softmax layer. This softmax gives the alignment scores. Its dimension will be the
# number of hidden states in the LSTM, i.e., 32 in this case. Taking its dot product along with the hidden states will
# provide the context vector:
# The get_config() method collects the input shape and other information about the model.

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()





def bilstm_attention(input_dim, number_classes,
                     optimizer='Adam',
                     lstm_layers=(128,64),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     initial_dropout_value=0.0,
                     dropout_rate=(0.3,0.2),
                     l1=1e-5, l2=1e-4,
                     dense_layers=(32,16),
                     dropout_rate_dense=(0.2,0.2),
                     dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, 21,), dtype='float32', name='main_input'))
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

        return model


x_train, x_test, x_dval, y_train, y_test, y_dval = \
    divide_dataset(fps_x_hot, fps_y_encoded, test_size=0.2, val_size=0.1)

vector_size = x_train.shape[1]
final_units = fps_y_hot.shape[1]

model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units)


dl=DeepML(x_train = x_train.astype(np.float32), y_train = y_train,x_test=x_test.astype(np.float32),
          y_test= y_test,
          number_classes=final_units, problem_type='multiclass',
          x_dval=x_dval.astype(np.float32), y_dval=y_dval, epochs=50, batch_size=64,
          path='..', report_name='attention',
          reduce_lr=True, early_stopping=True, checkpoint=True, tensorboard=False,
          early_stopping_patience=20, reduce_lr_patience=50, reduce_lr_factor=0.2, reduce_lr_min=0.00001,
          verbose=1)


dense = dl.run_model(model)
scores, report, cm, cm2 = dl.model_complete_evaluate()
model = dl.get_model()
dl.save_model('attention')
model.save('attention')

# very goodexplanation on attention
# https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

# https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e

# https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html