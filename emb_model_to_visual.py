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


dl_path = '/home/amsequeira/enzymeClassification/models/visualization/emb_500_post_5_ec90'
rp= '/post_pre_bilstm_attentio_emb5_lev1_ec90'
report_name = str(dl_path + rp)
model_name = str(dl_path)
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
        model.add(Embedding(input_dim=len(alphabet) + 1, output_dim=output_dim, mask_zero=True))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0), input_shape=(input_dim, 20,)))
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
    divide_dataset(fps_x, fps_y_encoded, test_size=0.2, val_size=0.2)
vector_size = x_train.shape[1]
final_units = fps_y_hot.shape[1]

model = KerasClassifier(build_fn=bilstm_attention_embedding, input_dim=vector_size, number_classes=final_units,
                        output_dim=5)

dl=DeepML(x_train = x_train.astype(np.float32), y_train = y_train,x_test=x_test.astype(np.float32),
          y_test= y_test,
          number_classes=final_units, problem_type='multiclass',
          x_dval=x_dval.astype(np.float32), y_dval=y_dval, epochs=100, batch_size=64,
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
model = dl.get_model()
model.model.save(model_name)

model = load_model(model_name)
model.summary()
x=0
emb_wei=None
emb_config=None
for layer in model.layers:
    print(layer.get_config(), layer.get_weights())
    if x ==1 :
        emb_wei = layer.get_weights()
        emb_config = layer.get_config()
    x+=1

emb_wei = pd.DataFrame(emb_wei[0])
emb_wei.to_csv(dl_path+'emb_wei')

# embeddings = model.layers[1].get_weights()[0]
# `embeddings` has a shape of (num_vocab, embedding_dim)

# `word_to_index` is a mapping (i.e. dict) from words to their index, e.g. `love`: 69
K.clear_session()
tf.keras.backend.clear_session()

