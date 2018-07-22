import configparser as ConfigParser
import pandas as pd
from extractor import Extractor
from feature import Feature
import numpy as np
import math
import random
from utils import LogUtil, DataUtil, MyUtil
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
_stemmer = SnowballStemmer('spanish')
import re
stops = set(stopwords.words("spanish"))

class FeatureAnalysis():

    def __init__(self, config_fp,data_set_name):
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), data_set_name)).fillna(value="")

    def checkFeature(self,num_features,num_sample,mul_feature_name):
        offline_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                            self.config.get('FEATURE', 'feature_selected_analysis').split(),
                                            self.config.get('FEATURE', 'online_rawset_name'),
                                            self.config.get('FEATURE', 'will_save'))
        for i in range(num_features):
            cur_feature = []
            for j in range(num_sample):
                cur_feature.append(offline_features[j,i])
                value = offline_features[j,i]
            X = np.asanyarray(cur_feature)
            if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum()) and not np.isfinite(X).all()):
                    print(value,i,j,mul_feature_name[i])
                    raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)
            if len(cur_feature) != num_sample:
                print("feature is not num_sample",i,mul_feature_name[i])
            print("feature ok",i,mul_feature_name[i])
            

    def getFeatureNumber(self):
        result = self.config.get('FEATURE', 'feature_selected_num_analysis').split()
        features_num = []
        for e in result:
            cur_feature_num = int(e.split('-')[1])
            features_num.append(cur_feature_num)
        return sum(features_num)
    
    def getFeatureName(self):
        result = self.config.get('FEATURE', 'feature_selected_num_analysis').split()
        features_num = []
        features_name = []
        multi_feature_name = []
        for e in result:
            cur_feature_name = e.split('-')[0]
            cur_feature_num = int(e.split('-')[1])
            features_num.append(cur_feature_num)
            features_name.append(cur_feature_name)
        for e,cur_feature_name in zip(features_num,features_name):
            for i in range(e):
                multi_feature_name.append(cur_feature_name)
        return multi_feature_name

    def getCalcPearson(self,num_features,mul_feature_name):
        offline_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                            self.config.get('FEATURE', 'feature_selected_analysis').split(),
                                            self.config.get('FEATURE', 'offline_rawset_name'),
                                            self.config.get('FEATURE', 'will_save'))
        label = []
        for index, row in self.data.iterrows():
            label.append(float(row['is_duplicate']))
        num_sample = len(label)
        calc_pearson_all = []
        
        for i in range(num_features):
            cur_feature = []
            for j in range(num_sample):
                cur_feature.append(offline_features[j,i])
            assert len(cur_feature) == num_sample, "cur_feature is not num_sample"
            cur_calc = self.calcPearson(label,cur_feature)
            calc_pearson_all.append(cur_calc)
            print(mul_feature_name[i],i,cur_calc)
        return calc_pearson_all
      
    def calcMean(self,x,y):
        x = np.array(x)
        y = np.array(y)
        return np.mean(x),np.mean(y)

    def saveNameAndCalc(self,save_filename,mul_feature_name,feature_calcpearson):
        feature_analysis_pt = self.config.get('DIRECTORY', 'feature_analysis_pt') + save_filename
        merge_string = []
        for cur_feature_name,cur_feature_calc in zip(mul_feature_name,feature_calcpearson):
            merge_string.append(str(cur_feature_name)+':'+str(cur_feature_calc))
        DataUtil.save_vector(feature_analysis_pt, merge_string, 'w')

    def calcPearson(self,x,y):
        x_mean,y_mean = self.calcMean(x,y)   #计算x,y向量平均值
        n = len(x)
        sumTop = 0.0
        sumBottom = 0.0
        x_pow = 0.0
        y_pow = 0.0
        for i in range(n):
            sumTop += (x[i]-x_mean)*(y[i]-y_mean)
        for i in range(n):
            x_pow += math.pow(x[i]-x_mean,2)
        for i in range(n):
            y_pow += math.pow(y[i]-y_mean,2)
        sumBottom = math.sqrt(x_pow*y_pow)
        p = sumTop/sumBottom
        return p

class Sign():
    def __init__(self, config_fp=None):
        print("wo shi initiala")
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))).fillna(value="")
        
    def getSign(self,sign_sentence):
        if sign_sentence == '!':
            return '!'
        elif sign_sentence == '?':
            return '?'
        else:
            return '.'

    def statisticDiffSign(self):
        num_all_sentence = 0
        num_all_excla_sentence = 0
        num_all_decla_sentence = 0
        num_all_interogative_sentence = 0
        double_excla_sentence = 0
        double_interogative_sentence = 0
        double_decla_sentence = 0
        double_excla_positive_sentence = 0
        double_excla_negative_sentence = 0
        double_interogative_positive_sentence = 0
        double_interogative_negative_sentence = 0
        double_decla_positive_sentence = 0
        double_decla_negative_sentence = 0

        single_excla_sentence = 0
        single_interogative_sentence = 0
        single_decla_sentence = 0
        single_excla_positive_sentence = 0
        single_excla_negative_sentence = 0
        single_interogative_positive_sentence = 0
        single_interogative_negative_sentence = 0
        single_decla_positive_sentence = 0
        single_decla_negative_sentence = 0


        for index, row in self.data.iterrows():
            english_sentence1 = str(row['english_sentence1'])
            english_sentence2 = str(row['english_sentence2'])
            is_duplicate = str(row['is_duplicate'])
            len_sentence1 = len(english_sentence1)
            len_sentence2 = len(english_sentence2)
            sign_sentence1 = self.getSign(english_sentence1[len_sentence1-1])
            sign_sentence2 = self.getSign(english_sentence2[len_sentence2-1])

            num_all_sentence += 1
            if sign_sentence1 == '!':
                num_all_excla_sentence += 1
            elif sign_sentence1 == '?':
                num_all_interogative_sentence += 1
            else:
                num_all_decla_sentence += 1

            if sign_sentence2 == '!':
                num_all_excla_sentence += 1
            elif sign_sentence2 == '?':
                num_all_interogative_sentence += 1
            else:
                num_all_decla_sentence += 1

            if sign_sentence1 == '!' and sign_sentence2 == '!':
                double_excla_sentence += 1
                if is_duplicate == '1':
                    double_excla_positive_sentence += 1
                else:
                    double_excla_negative_sentence += 1
            elif sign_sentence1 == '?' and sign_sentence2 == '?':
                double_interogative_sentence += 1
                if is_duplicate == '1':
                    double_interogative_positive_sentence += 1
                else:
                    double_interogative_negative_sentence += 1
            elif sign_sentence1 == '.' and sign_sentence2 == '.':
                double_decla_sentence += 1
                if is_duplicate == '1':
                    double_decla_positive_sentence += 1
                else:
                    double_decla_negative_sentence += 1

            if sign_sentence1 == '!' and sign_sentence2 != '!' or sign_sentence1 != '!' and sign_sentence2 == '!':
                single_excla_sentence += 1
                if is_duplicate == '1':
                    single_excla_positive_sentence += 1
                else:
                    single_excla_negative_sentence += 1
            if sign_sentence1 == '?' and sign_sentence2 != '?' or sign_sentence1 != '?' and sign_sentence2 == '?':
                single_interogative_sentence += 1
                if is_duplicate == '1':
                    single_interogative_positive_sentence += 1
                else:
                    single_interogative_negative_sentence += 1
            if sign_sentence1 == '.' and sign_sentence2 != '.' or sign_sentence1 != '.' and sign_sentence2 == '.':
                single_decla_sentence += 1
                if is_duplicate == '1':
                    single_decla_positive_sentence += 1
                else:
                    single_decla_negative_sentence += 1

        print(double_decla_sentence)
        print(double_interogative_sentence)
        print(double_excla_sentence)
        print(single_decla_sentence)
        print(single_interogative_sentence)
        print(single_excla_sentence)

        print("positive and negative")
        print(double_decla_positive_sentence)
        print(double_decla_negative_sentence)
        print(double_interogative_positive_sentence)
        print(double_interogative_negative_sentence)
        print(double_excla_positive_sentence)
        print(double_excla_negative_sentence)
        print(single_decla_positive_sentence)
        print(single_decla_negative_sentence)
        print(single_interogative_positive_sentence)
        print(single_interogative_negative_sentence)
        print(single_excla_positive_sentence)
        print(single_excla_negative_sentence)
        
        print("num_all")
        print(num_all_decla_sentence)
        print(num_all_interogative_sentence)
        print(num_all_excla_sentence)
        print(num_all_sentence)

    def getAllSign(self):
        sign_list = []
        for index, row in self.data.iterrows():
            if index > 2000000000:
                break
            english_sentence1 = str(row['english_sentence1'])
            english_sentence2 = str(row['english_sentence2'])
            is_duplicate = str(row['is_duplicate'])
            len_sentence1 = len(english_sentence1)
            len_sentence2 = len(english_sentence2)
            sign_sentence1 = english_sentence1[len_sentence1-1]
            sign_sentence2 = english_sentence2[len_sentence2-1]
            if sign_sentence1 not in sign_list and not(sign_sentence1.lower() >= 'a' and sign_sentence1.lower() <= 'z') and not (sign_sentence1.lower() >= '0' and sign_sentence1.lower() <= '9'):
                sign_list.append(sign_sentence1)
            if sign_sentence2 not in sign_list and not(sign_sentence2.lower() >= 'a' and sign_sentence2.lower() <= 'z') and not (sign_sentence2.lower() >= '0' and sign_sentence2.lower() <= '9'):
                sign_list.append(sign_sentence2)
        return sign_list

class NumDiffSentence():
    def __init__(self, config_fp=None):
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_test_a_20180516_csv'))).fillna(value="")
    
    def getDiffSentence(self):
        dul_num = {}
        for index, row in self.data.iterrows():
            q1 = str(row['spanish_sentence1']).strip()
            q2 = str(row.spanish_sentence2).strip()
            dul_num[q1] = dul_num.get(q1, 0) + 1
            if q1 != q2:
                dul_num[q2] = dul_num.get(q2, 0) + 1
        return dul_num

    def getDiffSentenceNum(self):
        dul_num = self.getDiffSentence()
        return len(dul_num)

    def getAllSentenceNum(self):
        return len(self.data) * 2

    def getMaxSameSentenceNum(self):
        max_num = -1
        dul_num = self.getDiffSentence()
        print(dul_num)
        for row in dul_num:
            if max_num < dul_num[row]:
                max_num = dul_num[row]
        return max_num

    def extendDataSet(self,save_pt):
        spanish_sentence1 = []
        spanish_sentence2 = []
        english_sentence1 = []
        english_sentence2 = []
        is_duplicateline = []

        for index, row in self.data.iterrows():
            if index <=  15000:
                continue
            if index % 100 == 0:
                print(index)
            for i in range(2):
                if i  == 0:
                    q1_spanish = str(row['spanish_sentence1']).strip()
                    q2_spanish = str(row['spanish_sentence2']).strip()
                    q1_english = str(row['english_sentence1']).strip()
                    q2_english = str(row['english_sentence2']).strip()
                else:
                    q1_spanish = str(row['spanish_sentence2']).strip()
                    q2_spanish = str(row['spanish_sentence1']).strip()
                    q1_english = str(row['english_sentence2']).strip()
                    q2_english = str(row['english_sentence1']).strip()
                for index1,cur_row in self.data.iterrows():
                    if index1 < index:
                        continue           
                    new_sample_is_duplicateline = int(cur_row['is_duplicate'])
                    sen1_spanish = cur_row['spanish_sentence1']
                    sen2_spanish = cur_row['spanish_sentence2']
                    if sen1_spanish == q1_spanish and new_sample_is_duplicateline == 1:
                        new_sample_sen1_english = cur_row['english_sentence2']
                        new_sample_sen2_english = q2_english
                        new_sample_sen1_spanish = sen2_spanish
                        new_sample_sen2_spanish = q2_spanish
                        new_sample_is_duplicateline = cur_row['is_duplicate']
                        spanish_sentence1.append(new_sample_sen1_spanish)
                        spanish_sentence2.append(new_sample_sen2_spanish)
                        english_sentence1.append(new_sample_sen1_english)
                        english_sentence2.append(new_sample_sen2_english)
                        is_duplicateline.append(new_sample_is_duplicateline)
                    
                    if sen2_spanish == q1_spanish and new_sample_is_duplicateline == 1:
                        new_sample_sen1_english = cur_row['english_sentence1']
                        new_sample_sen2_english = q2_english
                        new_sample_sen1_spanish = sen1_spanish
                        new_sample_sen2_spanish = q2_spanish
                        new_sample_is_duplicateline = cur_row['is_duplicate']

                        spanish_sentence1.append(new_sample_sen1_spanish)
                        spanish_sentence2.append(new_sample_sen2_spanish)
                        english_sentence1.append(new_sample_sen1_english)
                        english_sentence2.append(new_sample_sen2_english)
                        is_duplicateline.append(new_sample_is_duplicateline)
        data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicate':is_duplicateline})
        data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

    def filterExtendDataSet(self,extend_fp=None,save_fp=None):
        self.data = pd.read_csv(extend_fp).fillna(value="")
        merge_s1s2_set = set()
        spanish_sentence1 = []
        spanish_sentence2 = []
        english_sentence1 = []
        english_sentence2 = []
        is_duplicateline = []
        for index,row in self.data.iterrows():
            q1_spanish = str(row['spanish_sentence1'])
            q2_spanish = str(row['spanish_sentence2'])
            merge_q1q2_string = q1_spanish+q2_spanish
            merge_q2q1_string = q2_spanish+q1_spanish
            if merge_q1q2_string in merge_s1s2_set or merge_q2q1_string in merge_s1s2_set:
                continue
            else:
                merge_s1s2_set.add(merge_q1q2_string)
                merge_s1s2_set.add(merge_q2q1_string)
                new_sample_sen1_english = row['english_sentence1']
                new_sample_sen2_english = row['english_sentence2']
                new_sample_sen1_spanish = row['spanish_sentence1']
                new_sample_sen2_spanish = row['spanish_sentence2']
                new_sample_is_duplicateline = row['is_duplicate']
                spanish_sentence1.append(new_sample_sen1_spanish)
                spanish_sentence2.append(new_sample_sen2_spanish)
                english_sentence1.append(new_sample_sen1_english)
                english_sentence2.append(new_sample_sen2_english)
                is_duplicateline.append(new_sample_is_duplicateline)
        data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicate':is_duplicateline})
        data_frame.to_csv(save_fp,index=False,encoding='UTF-8')
        
    def getSamePairNum(self):
        samePairNum = 0
        allPairNum = len(self.data)
        for index, row in self.data.iterrows():
            q1 = str(row['english_sentence1']).strip()
            q2 = str(row.english_sentence2).strip()
            if q1 == q2:
                samePairNum += 1
        return 1.0 * samePairNum / allPairNum

    def bothSentenceInSameSubGraph(self,graph_result,sen1,sen2):
        for e in graph_result:
            cur_set = graph_result[e]
            if sen1 in cur_set and sen2 in cur_set:
                return True
        return  False
    
    def extendNegativeDataSet(self,save_pt=None,num_sample_generate=0,graph_result=None):
        spanish_sentence1 = []
        spanish_sentence2 = []
        english_sentence1 = []
        english_sentence2 = []
        is_duplicateline = []
        flag_num_same = 0
        for i in range(num_sample_generate):
            while True:
                random_index1 = int(random.random() * 21400)
                random_index2 = int(random.random() * 21400)
                row1 = self.data.iloc[random_index1]
                row2 = self.data.iloc[random_index2]                
                if row1['is_duplicate'] == 0 and row2['is_duplicate'] == 0:
                    new_sample_sen1_english = row1['english_sentence1']
                    new_sample_sen2_english = row2['english_sentence2']
                    new_sample_sen1_spanish = row1['spanish_sentence1']
                    new_sample_sen2_spanish = row2['spanish_sentence2']
                    new_sample_is_duplicateline = 0
                    if self.bothSentencesInSameSubGraph(graph_result,new_sample_sen1_spanish,new_sample_sen2_spanish):
                        flag_num_same+=1
                        print(flag_num_same)
                        print(new_sample_sen1_spanish)
                        print(new_sample_sen2_spanish)
                        continue
                    spanish_sentence1.append(new_sample_sen1_spanish)
                    spanish_sentence2.append(new_sample_sen2_spanish)
                    english_sentence1.append(new_sample_sen1_english)
                    english_sentence2.append(new_sample_sen2_english)
                    is_duplicateline.append(new_sample_is_duplicateline)
                    break
        data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicate':is_duplicateline})
        #data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

class SpanishChar():       
    def __init__(self, config_fp):
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))).fillna(value="")

    def getAllDiffChar(self):
        char_list = []
        for index, row in self.data.iterrows():
            spanish_sentence1 = str(row['spanish_sentence1'])
            spanish_sentence2 = str(row['spanish_sentence2'])
            sentence_sum = spanish_sentence1 + spanish_sentence2
            for e in sentence_sum:
                if e.lower() not in char_list:
                    char_list.append(e.lower())
        char_list.sort()
        return char_list

class Rate(Extractor):
    def __init__(self, config_fp,data_set_name):
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_ptO'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
    
    def isMultiSentence(self,sentence):
        sentence = sentence[0:len(sentence)-1]

        if '.' in sentence:
            return True
        return False

    def getMultiSentenceRate(self):
        both_multi_sentence_num = 0
        sigle_multi_sentence_num = 0
        total_multi_sentence_num = 0
        total_sentence = 0
        total_pair = 0
        for index, row in self.data.iterrows():
            total_pair += 1
            sen1 = str(row['spanish_sentence1'])
            sen2 = str(row['spanish_sentence2'])
            if self.isMultiSentence(sen1) and self.isMultiSentence(sen2):
                both_multi_sentence_num += 1
                total_multi_sentence_num += 1
                print(row)
            if self.isMultiSentence(sen1) and self.isMultiSentence(sen2) == False or self.isMultiSentence(sen2) and self.isMultiSentence(sen2) == False:
                sigle_multi_sentence_num += 1
                total_multi_sentence_num += 1
        total_sentence = total_pair * 2

        return both_multi_sentence_num,sigle_multi_sentence_num,total_multi_sentence_num,total_sentence,total_pair

    def getPostiveNumandRate(self):
        postive_num = 0
        total_num = 0
        postive_rate = 0.0
        for index, row in self.data.iterrows():
            label = str(row['is_duplicate'])
            if label == '1':
                postive_num += 1
            total_num += 1
        postive_rate = float(postive_num) / total_num
        return total_num,postive_num, postive_rate

    def get719TrainRate(self):
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_ptO'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
        bothInSameSubGraph_num = 0
        singleInSameSubGraph_num = 0
        noneInSameSubGraph_num = 0
        total_num = len(self.data)
        for index, row in self.data.iterrows():
            sen1 = str(row['spanish_sentence1'])
            sen2 = str(row['spanish_sentence2'])
            if MyUtil.bothSentencesInSameSubSet(sen1,sen2):
                bothInSameSubGraph_num += 1
            if MyUtil.singleSentencesInSameSubSet(sen1,sen2):
                singleInSameSubGraph_num += 1
            if MyUtil.noneSentencesInSameSubSet(sen1,sen2):
                noneInSameSubGraph_num += 1
        print("both rate ", 1.0 * bothInSameSubGraph_num / total_num)
        print("sigle rate ", 1.0 * singleInSameSubGraph_num / total_num)
        print("none rate ", 1.0 * noneInSameSubGraph_num / total_num)

    def get719TestTate(self):
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_ptO'), self.config.get('FILE_NAME', 'preprocessing_test_csv'))).fillna(value="")
        bothInSameSubGraph_num = 0
        singleInSameSubGraph_num = 0
        noneInSameSubGraph_num = 0
        total_num = len(self.data)
        for index, row in self.data.iterrows():
            sen1 = str(row['spanish_sentence1'])
            sen2 = str(row['spanish_sentence2'])
            if MyUtil.bothSentencesInSameSubSet(sen1,sen2):
                bothInSameSubGraph_num += 1
            if MyUtil.singleSentencesInSameSubSet(sen1,sen2):
                singleInSameSubGraph_num += 1
            if MyUtil.noneSentencesInSameSubSet(sen1,sen2):
                noneInSameSubGraph_num += 1
        print("both rate ", 1.0 * bothInSameSubGraph_num / total_num)
        print("sigle rate ", 1.0 * singleInSameSubGraph_num / total_num)
        print("none rate ", 1.0 * noneInSameSubGraph_num / total_num)

class Graph():
    def __init__(self, config_fp):
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_cleaning_ptO'), self.config.get('FILE_NAME', 'preprocessing_train_merge_csv'))).fillna(value="")
    
    def stemming(self,row):
        #print(row)
        line = [_stemmer.stem(word) for word in row.split()]
        #print(" ".join(line))
        return ' '.join(line)
    def removePunctuation(self,row):
        r = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        line = re.sub(r,'',row)
        return line

    def getSpecificWordByGraphResult(self,grapg_result):
        new_graph_result = {}
        for e in grapg_result:
            cur_set = grapg_result[e]
            num_sentences = len(cur_set)
            all_words_in_curSet = {}
            cur_list = []
            for s in cur_set:
                s = self.stemming(s)
                s = self.removePunctuation(s)
                words = s.lower().split()
                for cur_w in words:
                    if cur_w not in stops and len(cur_w)>2:
                        if cur_w not in all_words_in_curSet:
                            all_words_in_curSet[cur_w] = 1.0 / num_sentences
                        else:
                            all_words_in_curSet[cur_w] = all_words_in_curSet[cur_w] + 1.0/num_sentences
            for cur_w in all_words_in_curSet:
                if all_words_in_curSet[cur_w] > 0.4:
                    cur_list.append(cur_w)
            cur_list.append(num_sentences)
            new_graph_result[e] = cur_list
            print(cur_list)






    def findIndexInMap(self,sentence_set_map,q):
        index_list = []
        for cur_key in sentence_set_map:
            cur_set = sentence_set_map[cur_key]
            if q in cur_set:
                index_list.append(cur_key)
        return index_list

    def buildGraph(self):
        print("Building grapg...")
        #sentence_num_map = {}
        sentence_set_map = {}
        for index, row in self.data.iterrows():
            q1 = str(row['spanish_sentence1']).strip()
            q2 = str(row['spanish_sentence2']).strip()
            label = int(row['is_duplicate'])
            if label == 0:
                continue
            index_list1 = self.findIndexInMap(sentence_set_map,q1)
            index_list2 = self.findIndexInMap(sentence_set_map,q2)
            if len(index_list1) > 1 or len(index_list2) > 1:
                print("Wrong： index len more than 1")
            if len(index_list1) > 0 and len(index_list2) > 0:
                set1 = sentence_set_map[index_list1[0]]
                set2 = sentence_set_map[index_list2[0]]
                if len(set1) > 5 and len(set2) > 5:
                    continue
                merge_set = set1 | set2
                sentence_set_map.pop(index_list2[0])
                sentence_set_map[index_list1[0]] = merge_set
            elif len(index_list1) == 0 and len(index_list2) > 0:
                set2 = sentence_set_map[index_list2[0]]
                set2.add(q1)
                sentence_set_map[index_list2[0]] = set2
            elif len(index_list1) > 0 and len(index_list2) == 0:
                set1 = sentence_set_map[index_list1[0]]
                set1.add(q2)
                sentence_set_map[index_list1[0]] = set1
            else:
                new_set = set()
                new_set.add(q1)
                new_set.add(q2)
                len_sentence_set_map = len(sentence_set_map)
                sentence_set_map[len_sentence_set_map] = new_set
        print("total diffeent meaning sentance: ", len(sentence_set_map))
        return sentence_set_map

    def extendNegativeDataByGraph(self,save_pt,sentenceTag_sentence_map):
        spanish_sentence1 = []
        spanish_sentence2 = []
        english_sentence1 = []
        english_sentence2 = []
        is_duplicateline = []
        for i in range(200):
            max_cur_set_select_num = 3
            cur_curset_select_num1 = 0
            if i not in sentenceTag_sentence_map:
                continue
            cur_set1 = sentenceTag_sentence_map[i]
            for q1 in cur_set1:
                cur_curset_select_num1 += 1
                if cur_curset_select_num1 > max_cur_set_select_num:
                    break
                for j in range(i+1,200):
                    cur_curset_select_num2 = 0
                    if j not in sentenceTag_sentence_map:
                        continue
                    cur_set2 = sentenceTag_sentence_map[j]
                    for q2 in cur_set2:
                        cur_curset_select_num2+=1
                        if cur_curset_select_num2 > max_cur_set_select_num:
                            break
                        spanish_sentence1.append(q1)
                        spanish_sentence2.append(q2)
                        english_sentence1.append('null')
                        english_sentence2.append('null')
                        is_duplicateline.append(0)
        data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicate':is_duplicate})
        #print(data_frame)
        data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

    def getNumSentencesOfDifferentMeaningSample(self,sentenceTag_sentence_map):
        sentenceTag_num_map = {}
        for e in sentenceTag_sentence_map:
            cur_sentence_set = sentenceTag_sentence_map[e]
            sentenceTag_num_map[e] = len(cur_sentence_set)
        print(sorted(sentenceTag_num_map.items(), key=lambda d: d[1]))
        return sentenceTag_num_map


if __name__ == '__main__':
    #compute_calcpearson checkFeature generate_negative_data_by1 generate_negative_data_byGraph
    #computeGraphRate
    #getNum
    #graph
    #predictFeature
    Task_Porpose = ['predictFeature','compute_calcpearson']
    config_fp = '../conf/featwheel.conf'
    all_data_file_name = 'cikm_english_and_spanish_train_20180516.csv'
    save_filename = 'feature_calcpearson.txt' 
    extend_data_save_pt = './extend_data2.csv'
    extend_fp = './extend_data.csv'
    #save_fp = './extend_data_filtered.csv'
    save_pt = './extend_negative_data_graph.csv'
    extend_negative_save_pt = './extend_negative_data.csv'
    
    #NumDiffSentence(config_fp).extendDataSet(extend_data_save_pt)
    #NumDiffSentence(config_fp).filterExtendDataSet(extend_fp,save_fp)
    #grapg_result = Graph(config_fp).buildGraph()
    #print(Graph(config_fp).getNumSentencesOfDifferentMeaningSample(grapg_result))
    #Graph(config_fp).extendNegativeDataByGraph(save_pt,grapg_result)
    #Rate(config_fp,all_data_file_name).get719Rate(grapg_result[115])
    #print(Rate(config_fp,all_data_file_name).getMultiSentenceRate())
    if 'predictFeature' in Task_Porpose:
        config = ConfigParser.ConfigParser()
        config.read(config_fp)
        save_pt = '%s' % config.get('DIRECTORY','feature_pt')
        MyUtil.getPredictFeature('train','RandomForest.dat',save_pt)
        MyUtil.getPredictFeature('test','RandomForest.dat',save_pt)

    if 'graph' in Task_Porpose:
        grapg_result = Graph(config_fp).buildGraph()
        Graph(config_fp).getSpecificWordByGraphResult(grapg_result)

    if 'getNum' in Task_Porpose:
        print(NumDiffSentence(config_fp).getMaxSameSentenceNum())

    if 'computeGraphRate' in Task_Porpose:
        #grapg_result = Graph(config_fp).buildGraph()
        #print((grapg_result))
        Rate(config_fp,all_data_file_name).get719TestTate()
        Rate(config_fp,all_data_file_name).get719TrainRate()

    if  'generate_negative_data_by1' in Task_Porpose:
        graph_result = Graph(config_fp).buildGraph()
        NumDiffSentence(config_fp).extendNegativeDataSet(extend_negative_save_pt,30904,graph_result)

    if 'compute_calcpearson' in Task_Porpose:
        config_fp = '../conf/featwheel.conf'
        all_data_file_name = 'preprocessing_train_merge.csv'
        num_features = FeatureAnalysis(config_fp,all_data_file_name).getFeatureNumber()
        mul_feature_name = FeatureAnalysis(config_fp,all_data_file_name).getFeatureName()
        feature_calcpearson = FeatureAnalysis(config_fp,all_data_file_name).getCalcPearson(num_features,mul_feature_name)

    if 'checkFeature' in Task_Porpose:
        config_fp = '../conf/featwheel.conf'
        train_data = 'preprocessing_train_merge.csv'
        test_data = 'preprocessing_test.csv'
        num_sample_test = 5000
        num_sample_train = 83207
        num_features = FeatureAnalysis(config_fp,train_data).getFeatureNumber()
        FeatureAnalysis(config_fp,all_data_file_name).checkFeature(num_features,num_sample,mul_feature_name)
        FeatureAnalysis(config_fp,all_data_file_name).checkFeature(num_features,num_sample,mul_feature_name) 
    print("I am ok")
