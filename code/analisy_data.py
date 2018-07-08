import configparser as ConfigParser
import pandas as pd
from extractor import Extractor

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
    precess_file_name = 'preprocessing_test.csv'
    print(Rate(config_fp,precess_file_name).getPostiveNumandRate())
    print("I am ok")
