from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
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

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
#from model import BasicModel

#这个cnn类似于siamese模型，把两个句子分别通过同一个cnn模型得到句子的representation，然后
#计算representation的相似度，可以直接使用距离公式来计算，也可以把两个representation合并后用
#2层全连接来计算最后的match socre，本次方法用的后者

class ARCI:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        kernel_count = 16
        kernel_size = 15
        pool_size = 15
        dropout_rate = 0.3
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False)
        con1d_layer = Conv1D(kernel_count, kernel_size, padding='same')
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))
        q_embed = emb_layer(seq1)
        d_embed = emb_layer(seq2)
        q_conv1 = con1d_layer(q_embed)
        d_conv1 = con1d_layer(d_embed)
        q_pool1 = MaxPooling1D(pool_size=pool_size)(q_conv1)
        d_pool1 = MaxPooling1D(pool_size=pool_size)(d_conv1)
        pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])
        pool1_flat = Flatten()(pool1)
        pool1_flat_drop = Dropout(dropout_rate)(pool1_flat)
        dense = Dense(32,activation='relu')(pool1_flat_drop)
        dense = Dense(16,activation='relu')(dense)
        dense = Dense(8,activation='relu')(dense)
        pred = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[seq1, seq2], outputs=pred)
        ada = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['acc'])

        return model