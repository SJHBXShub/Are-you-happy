import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
import re
from gensim.models import Word2Vec
import gensim
from gensim.models.wrappers import FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from cnn_model import *
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors

Word2Vec = KeyedVectors.load_word2vec_format('../diff_traditional/source_data/wiki.es.vec', binary=False)
train = pd.read_csv('preprocessing_train.csv')
test = pd.read_csv('preprocessing_test.csv')
test['label'] = -1
test['test_id'] = range(len(test))

print ("去掉停用词...")
def spa_norm_word_list(s1):
    s1 = str(s1).lower().split()
    stop_words = stopwords.words('spanish')
    s2 = [w for w in s1 if w not in stop_words]
    if len(s2) < 3:
        s1 = [w for w in s1 if w in Word2Vec]
        return s1
    else:
        s2 = [w for w in s2 if w in Word2Vec]
        return s2

train['spa_qura1_list'] = train['spa_qura1'].apply(spa_norm_word_list)
train['spa_qura2_list'] = train['spa_qura2'].apply(spa_norm_word_list)

test['spa_qura1_list'] = test['spa_qura1'].apply(spa_norm_word_list)
test['spa_qura2_list'] = test['spa_qura2'].apply(spa_norm_word_list)

print("去掉停用词结束...")

print("生成embedding...")

spa_list = list(train['spa_qura1_list'])
spa_list.extend(list(train['spa_qura2_list']))
spa_list.extend(list(test['spa_qura1_list']))
spa_list.extend(list(test['spa_qura2_list']))

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 55
EMBEDDING_DIM = 300

# 生成token词典
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(spa_list)

texts_1 = list(train['spa_qura1_list'])
texts_2 = list(train['spa_qura2_list'])
test_texts_1 = list(test['spa_qura1_list'])
test_texts_2 = list(test['spa_qura2_list'])

# 将cnn预测的结果拿到traditional model里面去训练
pred_data1 = texts_1 + test_texts_1
pred_data2 = texts_2 + test_texts_2

# 得到词索引,相当于把每个sentence给序列化
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

pred_data1 = tokenizer.texts_to_sequences(pred_data1)
pred_data2 = tokenizer.texts_to_sequences(pred_data2)

# number of unique words
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# 长度不足max_length的补0
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

pred_data1 = pad_sequences(pred_data1, maxlen = MAX_SEQUENCE_LENGTH)
pred_data2 = pad_sequences(pred_data2, maxlen = MAX_SEQUENCE_LENGTH)

print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

# 把每个单词变成维度为EMBEDDING_DIM的向量,向量根据word2vec得到
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in Word2Vec:
        embedding_matrix[i] = Word2Vec[word]
# 查看哪些词没有词向量
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

label = np.array(list(train.label))
folds = KFold(n_splits = 5, shuffle = True, random_state = 123)
pred_results = []
re_weight = True
val_loss_result = []
for idx_train, idx_val in folds.split(data_1):
    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = label[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = label[idx_val]
    
    Cnn = CNN()
    model = Cnn.build_model(embedding_matrix, data_1.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    
    hist = model.fit([data_1_train, data_2_train],
                     labels_train, 
                     validation_data=([data_1_val, data_2_val], labels_val),
                     epochs=20,
                     batch_size=512,
                     shuffle=True, 
                     callbacks=[early_stopping],
                     verbose = 2)
    
    bst_val_score = min(hist.history['val_loss'])
    val_loss_result.append(bst_val_score)
    #model.load_weights("weights-improvement.hdf5")
    #preds = model.predict([test_data_1, test_data_2], batch_size=2048, verbose=2)
    preds = model.predict([pred_data1, pred_data2], batch_size=2048, verbose=2)
    pred_results.append(preds)


print (np.mean(val_loss_result))
res = (pred_results[0] + pred_results[1] + pred_results[2] +
     pred_results[3] + pred_results[4]) / 5
print (np.mean(pred_results))
res = pd.DataFrame(res)
#test['pred'] = res
#test[['pred']].to_csv('cnnresult.txt', index = None, header = None)
res.to_csv('cnnresult.dat',index = None,header = 'cnnresult')