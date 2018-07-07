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

class CNN(object):
    @staticmethod
    def build_model(self,emb_matrix, max_sequence_length):
        
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )
        
        # 1D convolutions that can iterate over the word vectors
        conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
        conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
        conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
        conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
        conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        # Run inputs through embedding
        emb1 = emb_layer(seq1)
        emb2 = emb_layer(seq2)

        # Run through CONV + GAP layers
        conv1a = conv1(emb1)
        glob1a = GlobalAveragePooling1D()(conv1a)
        conv1b = conv1(emb2)
        glob1b = GlobalAveragePooling1D()(conv1b)

        conv2a = conv2(emb1)
        glob2a = GlobalAveragePooling1D()(conv2a)
        conv2b = conv2(emb2)
        glob2b = GlobalAveragePooling1D()(conv2b)

        conv3a = conv3(emb1)
        glob3a = GlobalAveragePooling1D()(conv3a)
        conv3b = conv3(emb2)
        glob3b = GlobalAveragePooling1D()(conv3b)

        conv4a = conv4(emb1)
        glob4a = GlobalAveragePooling1D()(conv4a)
        conv4b = conv4(emb2)
        glob4b = GlobalAveragePooling1D()(conv4b)

        conv5a = conv5(emb1)
        glob5a = GlobalAveragePooling1D()(conv5a)
        conv5b = conv5(emb2)
        glob5b = GlobalAveragePooling1D()(conv5b)

        conv6a = conv6(emb1)
        glob6a = GlobalAveragePooling1D()(conv6a)
        conv6b = conv6(emb2)
        glob6b = GlobalAveragePooling1D()(conv6b)

        mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
        mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

        # We take the explicit absolute difference between the two sentences
        # Furthermore we take the multiply different entries to get a different measure of equalness
        diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
        mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
        mul_diff = Lambda(lambda x: x[0] - x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

        # # Add the magic features
        # magic_input = Input(shape=(5,))
        # magic_dense = BatchNormalization()(magic_input)
        # magic_dense = Dense(64, activation='relu')(magic_dense)

        # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
        # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
        # distance_input = Input(shape=(20,))
        # distance_dense = BatchNormalization()(distance_input)
        # distance_dense = Dense(128, activation='relu')(distance_dense)

        # Merge the Magic and distance features with the difference layer
        # merge = concatenate([diff, mul, magic_dense, distance_dense])
        
        #other_feature = Input(shape = (other_feature_len,))
        merge = concatenate([diff, mul, mul_diff])

        # The MLP that determines the outcome
        x = Dropout(0.2)(merge)
        x = BatchNormalization()(x)
        x = Dense(300, activation='relu')(x)

        #last_merge = concatenate([x, other_feature])
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        pred = Dense(1, activation='sigmoid')(x)

        # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        return model