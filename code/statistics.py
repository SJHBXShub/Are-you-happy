#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/20 14:28
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

import sys
from imp import reload
import math
import chardet
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import configparser as ConfigParser

from utils import NgramUtil, DistanceUtil, LogUtil, MathUtil
from utils import MISSING_VALUE_NUMERIC

from extractor import Extractor
from preprocessor import TextPreProcessor
import scipy

from scipy.stats import skew, kurtosis
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel

stops = set(stopwords.words("spanish"))
snowball_stemmer = SnowballStemmer('spanish')
reload(sys)

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
        not_cnt1 += q1_words.count(b'ni')
        not_cnt2 += q2_words.count(b'ni')
        not_cnt1 += q1_words.count(b'nunca')
        not_cnt2 += q2_words.count(b'nunca')

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

class TFCount(Extractor):

    def __init__(self, config):
        Extractor.__init__(self, config)
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(config)
        self.tf, self.tf_result = self.init_tf()

    def init_tf(self):
        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
        test_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_test_csv'))).fillna(value="")

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        stop_words=None)
        tf_txt = pd.Series(
            train_data['spanish_sentence1'].tolist() + train_data['spanish_sentence2'].tolist() + test_data['spanish_sentence1'].tolist() +
            test_data['spanish_sentence2'].tolist()).astype(str)
        tf_result = tf_vectorizer.fit_transform(tf_txt)
        LogUtil.log("INFO", "init TF done ")
        return tf_vectorizer, tf_result

class TFIDFWordMatchShare(Extractor):

    def __init__(self, config):
        Extractor.__init__(self, config)
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(config)

        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
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
        self.tfidf,self.tfidf_result = self.init_tfidf()

    def init_tfidf(self):
        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
        test_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_test_csv'))).fillna(value="")

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf_txt = pd.Series(
            train_data['spanish_sentence1'].tolist() + train_data['spanish_sentence2'].tolist() + test_data['spanish_sentence1'].tolist() +
            test_data['spanish_sentence2'].tolist()).astype(str)
        tfidf_result = tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")
        return tfidf, tfidf_result

    def extract_row(self, row):
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        fs = list()
        fs.append(np.sum(self.tfidf.transform([str(q1)]).data))
        fs.append(np.sum(self.tfidf.transform([str(q2)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q1)]).data))
        fs.append(np.mean(self.tfidf.transform([str(q2)]).data))
        fs.append(len(self.tfidf.transform([str(q1)]).data))
        fs.append(len(self.tfidf.transform([str(q2)]).data))
        return fs

    def get_feature_num(self):
        return 6

class DulNum(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.dul_num = self.generate_dul_num()

    def generate_dul_num(self):
        # load data set
        train_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
        test_data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), self.config.get('FILE_NAME', 'preprocessing_test_csv'))).fillna(value="")

        dul_num = {}
        for index, row in train_data.iterrows():
            q1 = str(row['spanish_sentence1']).strip()
            q2 = str(row.spanish_sentence2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        for index, row in test_data.iterrows():
            q1 = str(row.spanish_sentence1).strip()
            q2 = str(row.spanish_sentence2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        return dul_num


    def extract_row(self, row):
        q1 = str(row['spanish_sentence1']).strip()
        q2 = str(row['spanish_sentence2']).strip()

        dn1 = self.dul_num[q1]
        dn2 = self.dul_num[q2]
        return [dn1, dn2, max(dn1, dn2), min(dn1, dn2)]

    def get_feature_num(self):
        return 4

class EnCharCount(Extractor):
    def extract_row(self, row):
        s = 'abcdefghijklmnopqrstuvwxyzáéíñóú'

        q1 = str(row['spanish_sentence1']).strip().lower()
        q2 = str(row['spanish_sentence2']).strip().lower()
        fs1 = [0] * 33
        fs2 = [0] * 33
        for index in range(len(q1)):
            c = q1[index]
            if 0 <= s.find(c):
                fs1[s.find(c)] += 1
        for index in range(len(q2)):
            c = q2[index]
            if 0 <= s.find(c):
                fs2[s.find(c)] += 1
        return fs1 + fs2 + list(abs(np.array(fs1) - np.array(fs2)))

    def get_feature_num(self):
        return 33 * 3

class NgramJaccardCoef(Extractor):
    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence1'])))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence2'])))]
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
        return fs

    def get_feature_num(self):
        return 3

class NgramDiceDistance(Extractor):

    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence1'])))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence2'])))]
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
        return fs

    def get_feature_num(self):
        return 3

class Distance(Extractor):

    def __init__(self, config_fp, distance_mode):
        Extractor.__init__(self, config_fp)
        self.feature_name += '_%s' % distance_mode
        self.valid_distance_mode = ['edit_dist', 'compression_dist']
        assert distance_mode in self.valid_distance_mode, "Wrong aggregation_mode: %s" % distance_mode
        self.distance_mode = distance_mode
        self.distance_func = getattr(DistanceUtil, self.distance_mode)

    def extract_row(self, row):
        q1 = str(row['spanish_sentence1']).strip()
        q2 = str(row['spanish_sentence2']).strip()
        q1_stem = ' '.join([str(snowball_stemmer.stem(word).encode('utf-8')) for word in
                            nltk.word_tokenize(TextPreProcessor.clean_text(str(row['english_sentence1'])))])
        q2_stem = ' '.join([str(snowball_stemmer.stem(word).encode('utf-8')) for word in
                            nltk.word_tokenize(TextPreProcessor.clean_text(str(row['english_sentence2'])))])
        q1_stem = q1_stem.replace("b'","")
        q1_stem = q1_stem.replace("'","")
        q2_stem = q2_stem.replace("b'","")
        q2_stem = q2_stem.replace("'","")

        return [self.distance_func(q1, q2), self.distance_func(q1_stem, q2_stem)]

    def get_feature_num(self):
        return 2

class PhraseToSentenceDistance(Extractor):
    def __init__(self, config_fp, distance_mode):
        Extractor.__init__(self, config_fp)
        self.feature_name += '_%s' % distance_mode
        self.valid_distance_mode = ['edit_dist', 'compression_dist']
        assert distance_mode in self.valid_distance_mode, "Wrong aggregation_mode: %s" % distance_mode
        self.distance_mode = distance_mode
        self.distance_func = getattr(DistanceUtil, self.distance_mode)

    def extract_row(self,row):
        q1 = str(row['spanish_sentence1']).strip()
        q2 = str(row['spanish_sentence2']).strip()
        num_word1 = len(q1.split())
        num_word2 = len(q2.split())
        min_dis = 9999999999.0

        if num_word1 < 5 and (num_word2 - num_word1) > 4 or num_word2 < 5 and (num_word1 - num_word2) > 4:
            if num_word1 > num_word2:
                tem = q1
                q1 = q2
                q2 = tem
            for i in range(abs(num_word2 - num_word1)):
                cur_dis = self.distance_func(q1, ' '.join(q2.split()[i:i+num_word1]))
                if cur_dis < min_dis:
                    min_dis = cur_dis
            return [min_dis]
        else:
            return [self.distance_func(q1, q2)]

    def get_feature_num(self):
        return 1

class NgramDistance(Distance):

    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence1'])))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(TextPreProcessor.clean_text(str(row['spanish_sentence2'])))]

        fs = list()
        aggregation_modes_outer = ["mean", "max", "min", "median"]
        aggregation_modes_inner = ["mean", "std", "max", "min", "median"]
        for n_ngram in range(1, 5):
            q1_ngrams = NgramUtil.ngrams(q1_words, n_ngram)
            q2_ngrams = NgramUtil.ngrams(q2_words, n_ngram)

            val_list = list()
            for w1 in q1_ngrams:
                _val_list = list()
                for w2 in q2_ngrams:
                    s = self.distance_func(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[MISSING_VALUE_NUMERIC]]

            for mode_inner in aggregation_modes_inner:
                tmp = list()
                for l in val_list:
                    tmp.append(MathUtil.aggregate(l, mode_inner))
                fs.extend(MathUtil.aggregate(tmp, aggregation_modes_outer))
            return fs

    def get_feature_num(self):
        return 4 * 5

class PowerfulWord(object):
    def __init__(self, config_fp):
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)


    @staticmethod
    def load_powerful_word(fp):
        powful_word = []
        f = open(fp, 'r',encoding='utf-8')
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            powful_word.append((word, stats))
        f.close()
        return powful_word


    @staticmethod
    def generate_powerful_word(config, begin_indexs,end_index):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现在语句对的数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        conf = ConfigParser.ConfigParser()
        conf.read(config)
        data = pd.read_csv('%s/%s' % (conf.get('DIRECTORY', 'csv_spanish_cleaning_pt'), conf.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
        subset_indexs = []
        for i in range(begin_indexs,end_index):
            subset_indexs.append(i)
        words_power = {}
        train_subset_data = data.iloc[subset_indexs, :]
        for index, row in train_subset_data.iterrows():
            label = int(row['is_duplicate'])
            q1_words = str(row['spanish_sentence1']).lower().split()
            q2_words = str(row['spanish_sentence2']).lower().split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word in stops:
                    continue
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    #change_yj
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确数量
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确数量
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(subset_indexs)
            # 计算正确语句对比例
            if words_power[word][0] > 1e-6:
                words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][6], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power

    @staticmethod
    def save_powerful_word(words_power, fp):
        f = open(fp, 'w',encoding='utf-8')
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()

class PowerfulWordDoubleSide(Extractor):

    def __init__(self, config_fp, thresh_num=0, thresh_rate=0.0):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/%s.txt' % (
            self.config.get('DIRECTORY', 'devel_pt'), self.config.get('FILE_NAME', 'powerful_word_name'))
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_dside = PowerfulWordDoubleSide.init_powerful_word_dside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_dside(pword, thresh_num, thresh_rate):
        pword_dside = []
        pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
        pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
        pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
        # LogUtil.log('INFO', 'Double side power words(%d): %s' % (len(pword_dside), str(pword_dside)))
        return pword_dside

    def extract_row(self, row):
        tags = []
        q1_words = str(row['spanish_sentence1']).lower().split()
        q2_words = str(row['spanish_sentence2']).lower().split()
        for word in self.pword_dside:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    def get_feature_num(self):
        return len(self.pword_dside)

class PowerfulWordOneSide(Extractor):

    def __init__(self, config_fp, thresh_num=0, thresh_rate=0.0):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/%s.txt' % (
            self.config.get('DIRECTORY', 'devel_pt'), self.config.get('FILE_NAME', 'powerful_word_name'))
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_oside = PowerfulWordOneSide.init_powerful_word_oside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_oside(pword, thresh_num, thresh_rate):
        pword_oside = []
        pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)

        pword_oside.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
        LogUtil.log('INFO', 'One side power words(%d): %s' % (
            len(pword_oside), str(pword_oside)))
        return pword_oside

    def extract_row(self, row):
        tags = []
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        for word in self.pword_oside:
            if (word in q1_words) and (word not in q2_words):
                tags.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    def get_feature_num(self):
        return len(self.pword_oside)

class PowerfulWordDoubleSideRate(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/%s.txt' % (
            self.config.get('DIRECTORY', 'devel_pt'), self.config.get('FILE_NAME', 'powerful_word_name'))
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_row(self, row):
        num_least = 20
        power_double = 1.0
        tag = []
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        share_words = list(q1_words.intersection(q2_words))
        
        for word in share_words:
            if word not in self.pword_dict:
                continue
            if self.pword_dict[word][0] * self.pword_dict[word][5] < num_least:
                continue
            power_double *= math.pow(2,self.pword_dict[word][6])
        tag.append(power_double - 1.0)
        return tag

    def get_feature_num(self):
        return 1

class fuzz_QRatio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.QRatio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_WRatio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.WRatio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_partial_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.partial_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_partial_token_set_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.partial_token_set_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_partial_token_sort_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.partial_token_sort_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_token_set_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.token_set_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class fuzz_token_sort_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.token_sort_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1

class PowerfulWordOneSideRate(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)

        powerful_word_fp = '%s/%s.txt' % (
            self.config.get('DIRECTORY', 'devel_pt'), self.config.get('FILE_NAME', 'powerful_word_name'))
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_row(self, row):
        num_least = 20
        power_single = 1.0
        tag = []
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word not in self.pword_dict:
                continue
            if self.pword_dict[word][0] * self.pword_dict[word][3] < num_least:
                continue
            power_single *= math.pow(2,self.pword_dict[word][4])
        tag.append(power_single - 1.0)
        return tag

    def get_feature_num(self):
        return 1

class long_common_sequence(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        len1 = len(q1_words)
        len2 = len(q2_words)
        dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
        for i in range(len1):
            for j in range(len2):
                if q1_words[i] == q2_words[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return [dp[len1-1][len2-1] * 1.0 / (len1 + len2)]

    def get_feature_num(self):
        return 1

class long_common_prefix(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        len1 = len(q1_words)
        len2 = len(q2_words)
        max_prefix = 0
        min_len = min(len1, len2)
        for i in range(min_len):
            if q1_words[i] == q2_words[i]:
                max_prefix += 1
        return [max_prefix * 1.0 / (len1 + len2)]

    def get_feature_num(self):
        return 1

class long_common_suffix(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        len1 = len(q1_words)
        len2 = len(q2_words)
        q1_words.reverse()
        q2_words.reverse()
        max_prefix = 0
        min_len = min(len1, len2)
        for i in range(min_len):
            if q1_words[i] == q2_words[i]:
                max_prefix += 1
        return [max_prefix * 1.0 / (len1 + len2)]

    def get_feature_num(self):
        return 1

class long_common_substring(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        len1 = len(q1_words)
        len2 = len(q2_words)
        dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
        for i in range(len1):
            for j in range(len2):
                if q1_words[i] == q2_words[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = 0
        return [dp[len1-1][len2-1] * 1.0 / (len1 + len2)]

    def get_feature_num(self):
        return 1

class levenshtein_distance(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        s = q1_words
        t = q2_words
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)  
            for j in range(len(v0)):
                v0[j] = v1[j]  
        return [v1[len(t)] / (len(s) + len(t))]

    def get_feature_num(self):
        return 1

class cityblock_distance_ave_idf(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        words_pt = '%s/%s' % (self.config.get('DIRECTORY', 'source_pt'), self.config.get('FILE_NAME', 'words_txt'))
        self.words_dict = self.getWordsDict(words_pt=words_pt)
        print("get word2vec_dict...")
        word2vec_pt = '%s/%s' % (self.config.get('DIRECTORY', 'source_pt'), self.config.get('FILE_NAME', 'wiki_es_vec'))
        self.word2vec_dict = self.getWord2VecDict(word2vec_pt)
        print("get word2vec_dict done")
    
    def getWordsDict(self,words_pt):
        words_dict = {}
        with open(words_pt,'r') as f_in:
            for raw_line in f_in:
                line = raw_line.strip('\n').split()
                if len(line) != 2:
                    continue
                words_dict[line[0]] = float(line[1])
        return words_dict
    
    def getWord2VecDict(self,word2vec_pt):
        word2vec_dict = {}
        with open(word2vec_pt,'r') as f_in:
            for raw_line in f_in:
                line = raw_line.strip('\n\r').split()
                if line[0] in self.words_dict and line[1] not in self.words_dict:
                    #print(line[1:])
                    try:
                        word2vec_dict[line[0]] = [float(i) for i in line[1:]]
                    except:
                        continue
                    #word2vec_dict[line[0]] = [float(i) for i in line[1:]]
        return word2vec_dict

    def sent2vec_ave_idf(self,sen):
        M = [[ 0 for i in range(300) ]]
        words = sen
        for w in words:
            try:
                M.append([ self.words_dict[w] * x for x in self.word2vec_dict[w] ])
            except:
                continue
        M = np.array(M)
        num = M.shape[0]
        v = M.sum(axis=0)
        return v / num

    def extract_row(self, row):
        q1_words = str(row['spanish_sentence1']).lower().split()
        q2_words = str(row['spanish_sentence2']).lower().split()
        sent1_vectors = self.sent2vec_ave_idf(q1_words)
        sent2_vectors = self.sent2vec_ave_idf(q2_words)
        x,y = np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors)
        return [cityblock(x, y),jaccard(x, y),cosine(x, y),canberra(x, y),euclidean(x, y),braycurtis(x, y),minkowski(x, y, 3),skew(x),skew(y),kurtosis(x),kurtosis(y)]
    def get_feature_num(self):
        return 11

def demo():
    #Need_change
    config_fp = '../conf/featwheel.conf'
    #precess_file_name = 'preprocessing_train_merge.csv'
    precess_file_name = 'preprocessing_test.csv'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    devel_pt = config.get('DIRECTORY', 'devel_pt')
    fp_powerword = '%s/%s.txt' % (devel_pt,'words_power')
    begin_index = int(config.get('FEATURE', 'begin_index'))
    end_index = int(config.get('FEATURE', 'end_index'))
    TFIDF(config_fp).extract(precess_file_name)
    #cityblock_distance_ave_idf(config_fp).extract(precess_file_name)
   
    
    '''
    long_common_sequence(config_fp).extract(precess_file_name)
    long_common_prefix(config_fp).extract(precess_file_name)
    long_common_suffix(config_fp).extract(precess_file_name)
    long_common_substring(config_fp).extract(precess_file_name)
    levenshtein_distance(config_fp).extract(precess_file_name)
    fuzz_QRatio(config_fp).extract(precess_file_name)
    fuzz_WRatio(config_fp).extract(precess_file_name)
    fuzz_partial_ratio(config_fp).extract(precess_file_name)
    fuzz_partial_token_set_ratio(config_fp).extract(precess_file_name)
    fuzz_partial_token_sort_ratio(config_fp).extract(precess_file_name)
    fuzz_token_set_ratio(config_fp).extract(precess_file_name)
    fuzz_token_sort_ratio(config_fp).extract(precess_file_name)
    Not(config_fp).extract(precess_file_name)
    WordMatchShare(config_fp).extract(precess_file_name)
    TFIDFWordMatchShare(config_fp).extract(precess_file_name)
    Length(config_fp).extract(precess_file_name)
    LengthDiff(config_fp).extract(precess_file_name)
    LengthDiffRate(config_fp).extract(precess_file_name)
    TFIDF(config_fp).extract(precess_file_name)
    NgramJaccardCoef(config_fp).extract(precess_file_name)
    NgramDiceDistance(config_fp).extract(precess_file_name)
    EnCharCount(config_fp).extract(precess_file_name)
    DulNum(config_fp).extract(precess_file_name)
    
    NgramDistance(config_fp,'edit_dist').extract(precess_file_name)
    '''
    '''
    result = PowerfulWord.generate_powerful_word(config_fp,begin_index,end_index)
    PowerfulWord.save_powerful_word(result,fp_powerword)
    '''
    '''
    PowerfulWordOneSideRate(config_fp).extract(precess_file_name)
    PowerfulWordDoubleSideRate(config_fp).extract(precess_file_name)
    '''
    '''
    PowerfulWordDoubleSide(config_fp).extract(precess_file_name)
    PowerfulWordOneSide(config_fp).extract(precess_file_name)
    PowerfulWordDoubleSideRate(config_fp).extract(precess_file_name)
    PowerfulWordOneSideRate(config_fp).extract(precess_file_name)
    '''


if __name__ == '__main__':
    demo()
    '''
    q1 = "I am a student"
    q2 = "I love you and I am a student and you"
    print(q2.split()[0:2])
    if True:
        num_word1 = len(q1.split())
        num_word2 = len(q2.split())
        if num_word1 < 5 and (num_word2 - num_word1) > 4 or num_word2 < 5 and (num_word1 - num_word2) > 4:
            min_dis = 9999999999.0
            for i in range((num_word2 - num_word1)):
                print(q1, ' '.join(q2.split()[i:i+num_word1]))

                #cur_dis = self.distance_func(q1,q2.split()[i,i+num_word1])'''
                

