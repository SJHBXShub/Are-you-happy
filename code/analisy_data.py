import configparser as ConfigParser
import pandas as pd
from extractor import Extractor
from feature import Feature
import numpy as np
import math
from utils import LogUtil, DataUtil


class FeatureAnalysis():

    def __init__(self, config_fp,data_set_name):
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), data_set_name)).fillna(value="")

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
                X = np.asanyarray(value)
                if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum()) and not np.isfinite(X).all()):
                    print(value,i,j)
                    raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)
            else:
                print("feature ok")
            

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

    def getCalcPearson(self,num_features):
        offline_features = Feature.load_all(self.config.get('DIRECTORY', 'feature_pt'),
                                            self.config.get('FEATURE', 'feature_selected_analysis').split(),
                                            self.config.get('FEATURE', 'offline_rawset_name'),
                                            self.config.get('FEATURE', 'will_save'))
        label = []
        for index, row in self.data.iterrows():
            label.append(float(row['is_duplicateline']))
        num_sample = len(label)
        calc_pearson_all = []
        
        for i in range(num_features):
            cur_feature = []
            for j in range(num_sample):
                cur_feature.append(offline_features[j,i])
            print(len(cur_feature))
            print(num_sample)
            assert len(cur_feature) == num_sample, "cur_feature is not num_sample"
            cur_calc = self.calcPearson(label,cur_feature)
            calc_pearson_all.append(cur_calc)
            print("comput featur:",i,cur_calc)
        return calc_pearson_all
      
    def calcMean(self,x,y):
        sum_x = sum(x)
        sum_y = sum(y)
        n = len(x)
        x_mean = float(sum_x+0.0)/n
        y_mean = float(sum_y+0.0)/n
        return x_mean,y_mean

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
    def __init__(self, config_fp):
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
            is_duplicate = str(row['is_duplicateline'])
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
            is_duplicate = str(row['is_duplicateline'])
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
    def __init__(self, config_fp):
        print("wo shi initiala")
        # load configuration file
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), self.config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))).fillna(value="")
    
    def getDiffSentence(self):
        dul_num = {}
        for index, row in self.data.iterrows():
            q1 = str(row['english_sentence1']).strip()
            q2 = str(row.english_sentence2).strip()
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
        self.getAllData(data_set_name,config_fp)

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
            label = str(row['is_duplicateline'])
            if label == '1':
                postive_num += 1
            total_num += 1
        postive_rate = float(postive_num) / total_num
        return total_num,postive_num, postive_rate


if __name__ == '__main__':
    
    config_fp = '../conf/featwheel.conf'
    all_data_file_name = 'cikm_english_and_spanish_train_20180516.csv'
    save_filename = 'feature_calcpearson.txt' 
    num_sample = 5000


    #print(Rate(config_fp,all_data_file_name).getMultiSentenceRate())
    '''
    num_features = FeatureAnalysis(config_fp,all_data_file_name).getFeatureNumber()
    mul_feature_name = FeatureAnalysis(config_fp,all_data_file_name).getFeatureName()
    feature_calcpearson = FeatureAnalysis(config_fp,all_data_file_name).getCalcPearson(num_features)
    '''
    num_features = FeatureAnalysis(config_fp,all_data_file_name).getFeatureNumber()
    mul_feature_name = FeatureAnalysis(config_fp,all_data_file_name).getFeatureName()
    FeatureAnalysis(config_fp,all_data_file_name).checkFeature(num_features,num_sample,mul_feature_name)
    print("I am ok")
