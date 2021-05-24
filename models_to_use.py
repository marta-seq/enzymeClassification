import sys
import os
import logging
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D, MaxPool2D

from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Convolution2D, GRU, TimeDistributed, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

#input is 21 categorical 200 paded aa sequences
from keras.layers.merge import concatenate

def veltri_model(seq_len=700, final_units=8, output_dim = 256):
    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(seq_len,)))
        model.add(Embedding(input_dim=21, output_dim=output_dim, input_length=seq_len, mask_zero=True))
        model.add(Conv1D(
            filters=64,
            kernel_size=16,
            strides=1,
            padding='same',
            activation='relu'))
        model.add(MaxPool1D(pool_size=5, strides=1, padding='same'))
        model.add(LSTM(units=100,
                       dropout=0.1,
                       unroll=True,
                       return_sequences=False,
                       stateful=False))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(final_units, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

#input is physicochemical enzyme features of size [9920 X 1]
def deepen_model(vector_size, final_units ):
    with strategy.scope():
        model = Sequential()
        n_timesteps, n_features = 1, vector_size
        model.add(Input(shape=(n_timesteps,n_features)))
        model.add(Conv1D(
            filters=46,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'))
        model.add(MaxPool1D(pool_size=2, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(400, activation='sigmoid'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(final_units, activation = 'softmax'))
        print(model.summary())
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model



def deepec_model(vector_size, final_units):
    with strategy.scope():
        input =(Input(shape=(vector_size,21,1)))
        conv1=Conv2D(
            filters=128,
            kernel_size=(16,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv1 = MaxPool2D((4,4))(conv1)
        conv1=BatchNormalization()(conv1)
        conv2=Conv2D(
            filters=128,
            kernel_size=(8,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv2 = MaxPool2D((4,4))(conv2)
        conv2=BatchNormalization()(conv2)

        conv3 = Conv2D(
            filters=128,
            kernel_size=(4,21),
            strides=1,
            padding='same',
            activation='relu') (input)
        conv3 = MaxPool2D((4,4))(conv3)
        conv3=BatchNormalization()(conv3)

        concat =concatenate([conv1,conv2, conv3])
        flat = Flatten()(concat)
        dr = Dropout(0.5)(flat)
        hidden1 = Dense(512)(dr)
        # hidden2 = Dense(512)(hidden1)
        dr2 = Dropout(0.3)(hidden1)
        dr2=BatchNormalization()(dr2)

        output = Dense(final_units, activation = 'softmax')(dr)
        deepec = Model(inputs=input, outputs=output, name='deepec')
        print(deepec.summary())

        deepec.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return deepec




def daclstm_model(shape=(700,21),len_seq=700, final_units=8):
    # with strategy.scope():
    #just physico chemical features

    main_input = Input(shape=shape, name='main_input')
    # concat = main_input

    # design the deepaclstm model
    conv1 = Conv1D(42,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(main_input)
    conv1 = Reshape((len_seq,42, 1))(conv1)

    conv2 = Conv2D(42,3,1,activation='relu', padding='same', activity_regularizer=regularizers.l2(0.001))(conv1)
    conv2 = Reshape((len_seq,42*42))(conv2)
    conv2 = Dropout(0.5)(conv2)
    dense = Dense(400, activation='relu')(conv2)

    lstm1 = Bidirectional(LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5))(dense)
    lstm2 = Bidirectional(LSTM(units=300, return_sequences=True,recurrent_activation='sigmoid',dropout=0.5))(lstm1)

    # concat_features = Concatenate(axis=-1)([lstm_f2, lstm_b2, conv2_features])

    dr = Dropout(0.4)(lstm2)

    # concat_features = Flatten()(concat_features) #### add this part
    protein_features = Dense(600,activation='relu')(dr)
    fin = Flatten()(protein_features)
    # protein_features = TimeDistributed(Dense(600,activation='relu'))(concat_features)
    # protein_features = TimeDistributed(Dense(100,activation='relu', activity_regularizer=regularizers.l2(0.001)))(protein_features)
    main_output = Dense(final_units, activation='softmax', name='main_output')(fin)
    # main_output = (Dense(1, activation='sigmoid'))(protein_features)


    deepaclstm = Model(inputs=main_input, outputs=main_output)
    deepaclstm.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #loss= 'binary_crossentropy'
    print(deepaclstm.summary())
    return deepaclstm
