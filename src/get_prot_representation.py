
import sys
import os
import pandas as pd
import numpy as np
from tensorflow import keras
import operator
import random
from random import sample
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



########################################################################################################################
################################################## FOR X ###############################################################
########################################################################################################################
def hot_encoded_sequence(data, column_sequence, seq_len, alphabet, padding_truncating='post'):
    data = data[data[column_sequence].str.contains('!!!') == False]  # because 2 sequences in ecpred had !!!! to long to EXCEL... warning instead of sequence
    data = data[data.notna()]  # have some nas (should be taken out in further datasets)
    sequences = data[column_sequence].tolist()

    sequences_integer_ecoded = []
    for seq in sequences:
        if len(alphabet) < 25:  # alphabet x or alphabet normal
            seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
            seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
            seq3 = seq2.replace('U',
                                'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
            seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
            seq = seq4
            if len(alphabet) == 20:  # alphabet normal substitute every letters
                seq = seq4.replace('X', '')  # unknown character eliminated

        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)

    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    # pad sequences
    # todo pad in middle
    # pad inside the batch ?
    list_of_sequences_length = pad_sequences(sequences_integer_ecoded, maxlen=seq_len, dtype='int32',
                                             padding=padding_truncating, truncating=padding_truncating, value=0.0)

    # one hot encoding
    shape_hot = len(alphabet) * seq_len  # 20000
    encoded = to_categorical(list_of_sequences_length) # shape (samples, 1000,20)
    fps_x = encoded.reshape(encoded.shape[0], shape_hot)  # shape (samples, 20000)
    return data, fps_x, encoded

def hot_encoded_families(data, column_parameter='Cross-reference (Pfam)'):
    # divide parameter
    families = data[column_parameter]

    # remove columns with parameter NAn
    # data2 = data[data[column_parameter].notna()] # from 175267 to 167638

    fam = [i.split(';') for i in data[column_parameter]]  # split dos Pfam 'PF01379;PF03900;'
    fam = [list(filter(None, x)) for x in fam]  # remove '' empty string (because last ;
    # fam = [set(x) for x in fam]

    mlb = MultiLabelBinarizer()
    fam_ho = mlb.fit_transform(fam)
    classes = mlb.classes_
    len(classes)
    return data, fam_ho, classes

def physchemical(data):
    not_x = ['ec_number_ecpred', 'uniprot', 'sequence', 'uniref_90', 'ec_number']
    fps_x = data.drop(not_x, axis=1)
    columns = fps_x.columns
    return data, fps_x, columns

def nlf(data, seq_len, padding_truncating='post'):
    # transform nlf string to dataframe
    max_length = seq_len * 18 # size of parameters for each aa
    nlf = data['nlf']
    fps_x_encoded = []
    for line in nlf:
        line = [float(x) for x in line.split(',')]
        fps_x_encoded.append(line)

    # pad sequences
    fps_x_nlf = pad_sequences(fps_x_encoded, maxlen=max_length, padding=padding_truncating, truncating=padding_truncating, dtype='float32')
    return data, fps_x_nlf

def blosum(data, seq_len, padding_truncating='post'):
    max_length = seq_len * 24 # size of parameters for each aa
    blosum = data['blosum62']
    fps_x_encoded = []
    for line in blosum:
        line = [float(x) for x in line.split(',')]
        fps_x_encoded.append(line)

    # pad sequences
    fps_x_blosum=pad_sequences(fps_x_encoded, maxlen=max_length, padding=padding_truncating, truncating=padding_truncating, dtype='float32')
    return data, fps_x_blosum


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
    remake_sequence = []
    for seq in fps_x:
        integer_encoded = [int_to_char[int] for int in seq]
        remake_sequence.append(integer_encoded)
    return fps_x, remake_sequence


def deal_with_strange_aa(sequences, alphabet):
    new_sequences=[]
    for seq in sequences:
        if len(alphabet) < 25:  # alphabet x or alphabet normal
            seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
            seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
            seq3 = seq2.replace('U',
                                'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
            seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
            seq = seq4
            if len(alphabet) == 20:  # alphabet normal substitute every letters
                seq = seq4.replace('X', '')  # unknown character eliminated
        new_sequences.append(seq)
    return new_sequences