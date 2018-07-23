from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.layers.pooling import AveragePooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
import numpy as np 
import math
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from DynamicMaxPooling import *

#matchpyramid 模型是根据论文Text Matching as Image Recognition 实现的,作者将文本通过词与词之间的相似度
#构造成矩阵，然后用卷积提取矩阵的特征，最后得到match score，类似处理图像相似的cnn

class MatchPyramid:
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
        dpool_index = Input(name='dpool_index', shape = (max_sequence_length, max_sequence_length, 3), dtype='int32')
        # 2D convolutions
        conv2d = Conv2D(128, (5, 5), padding = 'same', activation='relu')
        #conv2d2 = Conv2D(128, (5, 5), padding = 'same', activation = 'relu')
        dpool = DynamicMaxPooling(5, 5)

        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        # Run inputs through embedding
        emb1 = emb_layer(seq1)
        emb2 = emb_layer(seq2)

        # use Dot construct sentence matrix
        cross = Dot(axes=[2, 2], normalize=False)([emb1, emb2])
        cross_reshape = Reshape((max_sequence_length, max_sequence_length, 1))(cross)

        # CNN process
        conv1 = conv2d(cross_reshape)
        #pool1 = AveragePooling2D(pool_size = (5,5))(conv1)
        pool1 = dpool([conv1, dpool_index])
#         conv2 = conv2d2(pool1)
#         pool2 = AveragePooling2D()(conv2)
        
        pool1_flat = Flatten()(pool1)
        pool1_flat_drop = Dropout(0.1)(pool1_flat)

        mlp = Dense(128, activation = 'relu')(pool1_flat_drop)
        mlp = Dropout(0.1)(mlp)
        mlp = Dense(64, activation = 'relu')(mlp)
        mlp = Dropout(0.1)(mlp)

        pred = Dense(1, activation = 'sigmoid')(mlp)

        model = Model(inputs=[seq1, seq2,dpool_index], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        return model