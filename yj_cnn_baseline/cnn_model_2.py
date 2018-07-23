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

#这个cnn类似于siamese模型，把两个句子分别通过同一个cnn模型得到句子的representation，然后
#计算representation的相似度，可以直接使用距离公式来计算，也可以把两个representation合并后用
#2层全连接来计算最后的match socre，本次方法用的后者
# epoch 150 0.31左右
class CNN2:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )

        # The distributing layer
        distr_layer = TimeDistributed(Dense(emb_matrix.shape[1], activation='relu'))

        # Lambda layer
        lamb_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(emb_matrix.shape[1], ))

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
        
        # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
        # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
        '''
        distance_input = Input(shape=(4,))
        distance_dense = BatchNormalization()(distance_input)
        distance_dense = Dense(200, activation='relu')(distance_dense)'''

        merged = concatenate([lamb1, lamb2])
        
        merged = Dropout(0.4)(merged)
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(50, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(20, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(4, activation='relu')(merged)
        merged = Dropout(0.4)(merged)
        merged = BatchNormalization()(merged)

        pred = Dense(1, activation='sigmoid')(merged)

        # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        ada = Adam(lr=0.0005)
        model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['acc'])

        return model