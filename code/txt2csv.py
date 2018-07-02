#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Author  : YangJ
import configparser as ConfigParser
import pandas as pd

def txtToCsv(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    while line:
        spanish_sentence1.append(line.split("\t")[1])
        english_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[3])
        english_sentence2.append(line.split("\t")[2])
        is_duplicateline.append(line.split("\t")[4][0])
        line = f.readline()
    data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')


if __name__ == '__main__':
    config_fp = 'C:/Users/jieyang/Desktop/GIT_code/kaggle-quora-question-pairs/conf/featwheel.conf'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    original_pt = '%s/%s' % (config.get('DIRECTORY', 'origin_spanish_pt'),config.get('FILE_NAME', 'cikm_english_train_20180516_txt'))
    save_pt = '%s/%s' % (config.get('DIRECTORY', 'csv_spanish_pt'),config.get('FILE_NAME', 'cikm_english_train_20180516_csv'))
    txtToCsv(original_pt, save_pt)