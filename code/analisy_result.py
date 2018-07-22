
import configparser as ConfigParser
import pandas as pd
from extractor import Extractor
import math
import numpy as np
import getopt
import sys
from utils import DataUtil
from feature import Feature

class PostProcessor(object):
    @staticmethod
    def adj(x, te, tr):
        a = te / tr
        b = (1 - te) / (1 - tr)
        return a * x / (a * x + b * (1 - x))
    @staticmethod
    def getResultMean(online_preds=None,online_preds_fp=None):
        if online_preds_fp:
            online_preds = DataUtil.load_vector(online_preds_fp, 'float')
        if len(online_preds) == 0:
            print("wrong,file len is 0")
        return sum(online_preds)/len(online_preds)

    @staticmethod
    def rescale(online_preds_fp):
        online_preds = DataUtil.load_vector(online_preds_fp, 'float')
        print(PostProcessor.getResultMean(online_preds))
        for index in range(len(online_preds)):
            score = online_preds[index]
            score = PostProcessor.adj(score, te=0.5, tr=0.25)
            online_preds[index] = score
        print(PostProcessor.getResultMean(online_preds))
        DataUtil.save_vector(online_preds_fp + '.rescale', online_preds, 'w')

class FileProcess():
    def __init__(self, config_fp,data_set_name):
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.data = pd.read_csv('%s/%s' % (self.config.get('DIRECTORY', 'csv_spanish_pt'), data_set_name)).fillna(value="")

    def mergePredictAndReal(self,index_fp,predict_fp,save_pt):
        f_index = open(index_fp,"r",encoding='UTF-8')
        f_predict = open(predict_fp,"r",encoding='UTF-8')
        line_index = f_index.readline()
        line_predict = f_predict.readline()
        english_sentence1 = []
        english_sentence2 = []
        spanish_sentence1 = []
        spanish_sentence2 = []
        predict_result = []
        real_result = []
        id_ = []
        loss = []
        while line_index and line_predict:
            index_num = int(line_index)
            predict_num = float(line_predict)
            row = self.data.loc[index_num]
            cur_label = int(row['is_duplicateline'])
            curr_loss = - cur_label * math.log(predict_num) - (1. - cur_label) * math.log(1 - predict_num)
            english_sentence1.append(row['english_sentence1'])
            spanish_sentence1.append(row['spanish_sentence1'])
            spanish_sentence2.append(row['spanish_sentence2'])
            english_sentence2.append(row['english_sentence2'])
            id_.append(row['id'])
            predict_result.append(row['is_duplicateline'])
            loss.append(curr_loss)
            real_result.append(predict_num)
            line_index = f_index.readline()
            line_predict = f_predict.readline()
        data_frame = pd.DataFrame({'id':id_,'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'zreal_result':real_result, 'zpredict_result':predict_result, 'zloss':loss})
        data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

if __name__ == '__main__':
    config_fp = '../conf/featwheel.conf'
    all_data_file_name = 'cikm_english_and_spanish_train_20180516.csv'
    index_fp = './se_valid.preprocessing_train_merge.csv.index'
    predict_fp = './se_valid.preprocessing_train_merge.csv.pred'
    save_pt = './result_analyse.csv'
    online_preds_fp = './200_iter.txt'
    #print(FileProcess(config_fp,all_data_file_name).mergePredictAndReal(index_fp,predict_fp,save_pt))
    PostProcessor().rescale(online_preds_fp)
    print("I am ok")