import configparser as ConfigParser
import pandas as pd
import numpy as np
import math
import nltk
from utils import LogUtil, DataUtil, MyUtil

class Word2Vec(object):
    @staticmethod
    def getWord2VecDict(word2vec_pt):
        print("get Word2Vec...")
        word2vec_dict = {}
        with open(word2vec_pt,'r',encoding='utf-8') as f_in:
            for raw_line in f_in:
                line = raw_line.strip('\n\r').split()
                if True:
                    try:
                        correct_flag = [float(i) for i in line[1:]]
                        word2vec_dict[line[0]] =  line[1:]
                        #print(word2vec_dict[line[0]])
                    except:
                        continue
            print("Word2Vec done")
        return word2vec_dict

    @staticmethod
    def getSmallWord2Vec(path_word2vec=None,path_train_data=None,path_test_data=None):
        word2vec_dict = Word2Vec.getWord2VecDict(path_word2vec)
        #word2vec_dict = {}
        small_word2vec_dict = {}
        not_found_word2vec = ' '.join(['0.0'] * 300)
        if path_train_data != None:
            data = pd.read_csv(path_train_data,encoding='UTF-8').fillna(value="")
            for index, row in data.iterrows():
                q1_words = nltk.word_tokenize(str(row['spanish_sentence1']).lower())
                q2_words = nltk.word_tokenize(str(row['spanish_sentence2']).lower())
                sen = q1_words + q2_words
                for word in sen:
                    if word not in small_word2vec_dict:
                        if len(word.split()) > 1:
                            print("I am wrong more than 1 word",word)
                        try:
                            small_word2vec_dict[word] = ' '.join(word2vec_dict[word])

                        except:
                            print("not found word", word)
                            small_word2vec_dict[word] = not_found_word2vec = not_found_word2vec
        if path_test_data != None:
            data = pd.read_csv(path_test_data).fillna(value="")
            for index, row in data.iterrows():
                q1_words = nltk.word_tokenize(str(row['spanish_sentence1']).lower())
                q2_words = nltk.word_tokenize(str(row['spanish_sentence2']).lower())
                sen = q1_words + q2_words
                for word in sen:
                    if word not in small_word2vec_dict:
                        if len(word.split()) > 1:
                            print("I am wrong more than 1 word",word)
                        try:
                            small_word2vec_dict[word] = ' '.join(word2vec_dict[word])
                        except:
                            print("not found word", word)
                            small_word2vec_dict[word] = not_found_word2vec
        return small_word2vec_dict

    @staticmethod
    def saveSmallWord2Vec(path_save, small_word2vec_dict):
        word2vec_list = []
        count = 0
        for key in small_word2vec_dict:
            count += 1
            if count < 20:
                print(key)
            cur_string = str(key) + ' ' + str(small_word2vec_dict[key])
            word2vec_list.append(cur_string)
        DataUtil.save_vector(path_save, word2vec_list, 'w')
    
    @staticmethod
    def changeWord2index(path_word2vec=None,path_word_dic=None,path_save=None):
        file_word_dic = open(path_word_dic,'r',encoding='utf-8')
        file_word2vec = open(path_word2vec,'r',encoding='utf-8')
        not_found_word2vec = ' '.join(['0.0'] * 300)
        word2vec_list = []
        line = file_word_dic.readline()
        word2index_dic = {}
        word2vec_dic = {}
        while line:
            cur_word = line.split()[0]
            cur_index = line.strip('\n\r').split()[1]
            word2index_dic[cur_word] = cur_index
            line = file_word_dic.readline()

        line = file_word2vec.readline()
        while line:
            #print(line)
            line = line.strip('\n\r').split()
            cur_key = str(line[0])
            cur_value = ' '.join(line[1:])
            word2vec_dic[cur_key] = cur_value
            line = file_word2vec.readline()

        for cur_key in word2index_dic:
            try:
                cur_value = word2vec_dic[cur_key]
            except:
                cur_value = not_found_word2vec
                print(cur_key,word2index_dic[cur_key])
            cur_string = str(word2index_dic[cur_key]) + ' ' + cur_value
            word2vec_list.append(cur_string)
        DataUtil.save_vector(path_save, word2vec_list, 'w')





if __name__ == '__main__':
    config_fp = '../conf/featwheel.conf'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    path_train_data = '%s/%s' % (config.get('DIRECTORY', 'csv_spanish_cleaning_pt1'), 'preprocessing_train_merge.csv')
    path_test_data = '%s/%s' % (config.get('DIRECTORY', 'csv_spanish_cleaning_pt1'), 'preprocessing_test.csv')
    path_word2vec = '%s/%s' % (config.get('DIRECTORY', 'source_pt'), config.get('FILE_NAME', 'wiki_es_vec'))
    path_save_word = './embed_glove_d300_norm_word'
    path_word_dic = './word_dict.txt'
    path_save_index = './embed_glove_d300_norm'
    path_word2vec = './embed_glove_d300_norm_word'
    #small_word2vec_dict = Word2Vec.getSmallWord2Vec(path_word2vec,path_train_data,path_test_data)
    #Word2Vec.saveSmallWord2Vec(path_save_word,small_word2vec_dict)
    Word2Vec.changeWord2index(path_word2vec=path_word2vec, path_word_dic = path_word_dic, path_save = path_save_index)
    print("I am OK")




    