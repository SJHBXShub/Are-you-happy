#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Author  : YangJ
import configparser as ConfigParser
import pandas as pd

def txtToCsvTrainEnglishVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    while line:
        english_sentence1.append(line.split("\t")[0])
        spanish_sentence1.append(line.split("\t")[1])
        english_sentence2.append(line.split("\t")[2])
        spanish_sentence2.append(line.split("\t")[3])
        

        is_duplicateline.append(line.split("\t")[4][0])
        line = f.readline()
    data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTrainSpanishVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    while line:
        english_sentence1.append(line.split("\t")[1])
        spanish_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[2])
        english_sentence2.append(line.split("\t")[3])

        is_duplicateline.append(line.split("\t")[4][0])
        line = f.readline()
    data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTestVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    while line:
        spanish_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[1])

        line = f.readline()
    data_frame = pd.DataFrame({'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTrainUnlabelVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    while line:
        english_sentence1.append(line.split("\t")[1])
        spanish_sentence1.append(line.split("\t")[0])
        line = f.readline()
    data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'spanish_sentence1':spanish_sentence1})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

if __name__ == '__main__':
    config_fp = 'C:/Users/jieyang/Desktop/GIT_code/Are-you-happy/conf/featwheel.conf'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    original_pt = '%s/%s' % (config.get('DIRECTORY', 'origin_spanish_pt'),config.get('FILE_NAME', 'cikm_unlabel_spanish_train_20180516_txt'))
    save_pt = '%s/%s' % (config.get('DIRECTORY', 'csv_spanish_pt'),config.get('FILE_NAME', 'cikm_unlabel_spanish_train_20180516_csv'))
    txtToCsvTestVersion(original_pt, save_pt)