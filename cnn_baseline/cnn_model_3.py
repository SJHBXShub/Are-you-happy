from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.core import SpatialDropout1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import numpy as np 
import math
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

#这个deep model先对sentence pair分别做一个embedding处理，LSTMs 处理，和带词向量的1DConv处理，再把
#三者处理的结果concatenate，再做dnn

class CNN3:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        ############# Embedding Process ############
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )

        lstm_emb_layer = Embedding(
            input_dim = emb_matrix.shape[0],
            output_dim = emb_matrix.shape[1],
            input_length = max_sequence_length
        )

        # The distributing layer
        distr_layer = TimeDistributed(Dense(emb_matrix.shape[1], activation='relu'))

        # Lambda layer
        lamb_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(emb_matrix.shape[1], ))

        ############# LSTMs Process #################
        lstm_layer = LSTM(emb_matrix.shape[1], dropout=0.2, recurrent_dropout=0.2)

        ############# 1DConv Process #################
#         conv1 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu')
#         conv2 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu')

        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        # Run inputs through embedding
        emb1 = emb_layer(seq1)
        emb2 = emb_layer(seq2)

        # Run through distributing layer
        dis1 = distr_layer(emb1)
        dis2 = distr_layer(emb2)

        # through lambda layer
        lamb1 = lamb_layer(dis1)
        lamb2 = lamb_layer(dis2)

        #1DCONV
#         conv1a = conv1(emb1)
#         conv1a = Dropout(0.2)(conv1a)
#         conv1a = conv2(conv1a)
#         conv1a = GlobalMaxPooling1D()(conv1a)
#         conv1a = Dropout(0.2)(conv1a)
#         conv1a = Dense(300)(conv1a)
#         conv1a = Dropout(0.2)(conv1a)
#         conv1a = BatchNormalization()(conv1a)

#         conv1b = conv1(emb2)
#         conv1b = Dropout(0.2)(conv1b)
#         conv1b = conv2(conv1b)
#         conv1b = GlobalMaxPooling1D()(conv1b)
#         conv1b = Dropout(0.2)(conv1b)
#         conv1b = Dense(300)(conv1b)
#         conv1b = Dropout(0.2)(conv1b)
#         conv1b = BatchNormalization()(conv1b)

        # LSTM
        lstm1 = lstm_emb_layer(seq1)
        lstm1 = Dropout(0.2)(lstm1)
        lstm1 = lstm_layer(lstm1)
        lstm2 = lstm_emb_layer(seq2)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = lstm_layer(lstm2)

        mergea = concatenate([lamb1,lstm1])
        mergeb = concatenate([lamb2,lstm2])

        merged = concatenate([mergea, mergeb])
        merged = BatchNormalization()(merged)

        merged = Dense(300, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(300, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(300, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(300, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)

        pred = Dense(1, activation='sigmoid')(merged)

        # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        return model