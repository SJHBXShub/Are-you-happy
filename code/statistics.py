#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/20 14:28
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


import math

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import configparser as ConfigParser

from utils import NgramUtil, DistanceUtil, LogUtil, MathUtil
from utils import MISSING_VALUE_NUMERIC

from extractor import Extractor
from preprocessor import TextPreProcessor

stops = set(stopwords.words("spanish"))
snowball_stemmer = SnowballStemmer('spanish')

class Not(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.snowball_stemmer = SnowballStemmer('spanish')

    def get_feature_num(self):
        return 5

    def extract_row(self, row):
        q1 = str(row['spanish_sentence1']).strip()
        q2 = str(row['spanish_sentence2']).strip()

        q1_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q1))]
        q2_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(q2))]
        
        not_cnt1 = q1_words.count(b'no')
        not_cnt2 = q2_words.count(b'no')

        fs = list()
        fs.append(not_cnt1)
        fs.append(not_cnt2)
        if not_cnt1 > 0 and not_cnt2 > 0:
            fs.append(1.)
        else:
            fs.append(0.)
        if (not_cnt1 > 0) or (not_cnt2 > 0):
            fs.append(1.)
        else:
            fs.append(0.)
        if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
            fs.append(1.)
        else:
            fs.append(0.)

        return fs

class WordMatchShare(Extractor):

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['spanish_sentence1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['spanish_sentence2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            return [0.]
        else:
            return [1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol]

    def get_feature_num(self):
        return 1

class TFIDFWordMatchShare(Extractor):

    def __init__(self, config):
        Extractor.__init__(self, config)
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(config)

        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))).fillna(value="")
        self.idf = TFIDFWordMatchShare.init_idf(train_data)

    @staticmethod
    def init_idf(data):
        idf = {}
        q_set = set()
        for index, row in data.iterrows():
            q1 = str(row['spanish_sentence1'])
            q2 = str(row['spanish_sentence2'])
            if q1 not in q_set:
                q_set.add(q1)
                words = q1.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
            if q2 not in q_set:
                q_set.add(q2)
                words = q2.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        LogUtil.log("INFO", "idf calculation done, len(idf)=%d" % len(idf))
        return idf

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['spanish_sentence1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['spanish_sentence2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        sum_shared_word_in_q1 = sum([q1words[w] * self.idf.get(w, 0) for w in q1words if w in q2words])
        sum_shared_word_in_q2 = sum([q2words[w] * self.idf.get(w, 0) for w in q2words if w in q1words])
        sum_tol = sum(q1words[w] * self.idf.get(w, 0) for w in q1words) + sum(
            q2words[w] * self.idf.get(w, 0) for w in q2words)
        if 1e-6 > sum_tol:
            return [0.]
        else:
            return [1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol]

    def get_feature_num(self):
        return 1

class Length(Extractor):
    def extract_row(self, row):
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        fs = list()
        fs.append(len(q1))
        fs.append(len(q2))
        fs.append(len(q1.split()))
        fs.append(len(q2.split()))
        return fs

    def get_feature_num(self):
        return 4

class LengthDiff(Extractor):
    def extract_row(self, row):
        q1 = row['spanish_sentence1']
        q2 = row['spanish_sentence2']
        return [abs(len(q1) - len(q2))]

    def get_feature_num(self):
        return 1

class LengthDiffRate(Extractor):
    def extract_row(self, row):
        len_q1 = len(row['spanish_sentence1'])
        len_q2 = len(row['spanish_sentence2'])
        if max(len_q1, len_q2) < 1e-6:
            return [0.0]
        else:
            return [1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)]

    def get_feature_num(self):
        return 1

class TFIDF(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.tfidf = self.init_tfidf()

    def init_tfidf(self):
        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))).fillna(value="")        
        test_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_test_a_20180516_csv'))).fillna(value="")

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(
            train_data['spanish_sentence1'].tolist() + train_data['spanish_sentence2'].tolist() + test_data['spanish_sentence1'].tolist() +
            test_data['spanish_sentence2'].tolist()).astype(str)
        tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")
        return tfidf

    def extract_row(self, row):
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        fs = list()
        fs.append(np.sum(self.tfidf.transform([str(q1)]).data))
        print("q1",str(q1))
        print(self.tfidf.transform([str(q1)]).data)
        fs.append(np.sum(self.tfidf.transform([str(q2)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q1)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q2)]).data))
        fs.append(len(self.tfidf.transform([str(q1)]).data))
        fs.append(len(self.tfidf.transform([str(q2)]).data))
        return fs

    def get_feature_num(self):
        return 6

def demo():
    #Need_change
    config_fp = '../conf/featwheel.conf'
    Length(config_fp).extract('cikm_english_train_20180516.csv')


if __name__ == '__main__':
    demo()
