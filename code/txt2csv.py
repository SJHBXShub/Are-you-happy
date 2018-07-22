#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Author  : YangJ
import configparser as ConfigParser
import pandas as pd

def txtToCsvTrainEnglishandSpanishVersion(original_english_pt,original_spanish_pt,save_pt):
    f_english = open(original_english_pt,"r",encoding='UTF-8')  
    f_spanish = open(original_spanish_pt,"r",encoding='UTF-8')
    line = f_english.readline() 

    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    id_ = []
    qid1 = []
    qid2 = []
    index = 0
    while line:
        english_sentence1.append(line.split("\t")[0])
        spanish_sentence1.append(line.split("\t")[1])
        english_sentence2.append(line.split("\t")[2])
        spanish_sentence2.append(line.split("\t")[3])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        is_duplicateline.append(line.split("\t")[4][0])
        line = f_english.readline()

    line = f_spanish.readline()
    while line:
        if index % 100 == 0:
            print(index)
        english_sentence1.append(line.split("\t")[1])
        spanish_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[2])
        english_sentence2.append(line.split("\t")[3])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        is_duplicateline.append(line.split("\t")[4][0])
        line = f_spanish.readline()

    #data_frame = pd.DataFrame({'id':id_,'qid1':qid1,'qid2':qid2,'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame = pd.DataFrame({'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')


def txtToCsvTrainEnglishVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    id_ = []
    qid1 = []
    qid2 = []
    index = 0
    while line:
        english_sentence1.append(line.split("\t")[0])
        spanish_sentence1.append(line.split("\t")[1])
        english_sentence2.append(line.split("\t")[2])
        spanish_sentence2.append(line.split("\t")[3])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        is_duplicateline.append(line.split("\t")[4][0])
        line = f.readline()
    data_frame = pd.DataFrame({'id':id_,'qid1':qid1,'qid2':qid2,'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTrainSpanishVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    id_ = []
    qid1 = []
    qid2 = []
    index = 0
    while line:
        english_sentence1.append(line.split("\t")[1])
        spanish_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[2])
        english_sentence2.append(line.split("\t")[3])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        is_duplicateline.append(line.split("\t")[4][0])
        line = f.readline()
    data_frame = pd.DataFrame({'id':id_,'qid1':qid1,'qid2':qid2,'english_sentence1':english_sentence1, 'english_sentence2':english_sentence2, 'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicateline':is_duplicateline})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTestVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    id_ = []
    qid1 = []
    qid2 = []
    index = 0
    while line:
        spanish_sentence1.append(line.split("\t")[0])
        spanish_sentence2.append(line.split("\t")[1])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        line = f.readline()
    data_frame = pd.DataFrame({'id':id_,'qid1':qid1,'qid2':qid2,'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')

def txtToCsvTrainUnlabelVersion(original_pt,save_pt):
    f = open(original_pt,"r",encoding='UTF-8')  
    line = f.readline() 
    spanish_sentence1 = []
    spanish_sentence2 = []
    english_sentence1 = []
    english_sentence2 = []
    is_duplicateline = []
    id_ = []
    qid1 = []
    qid2 = []
    index = 0
    while line:
        english_sentence1.append(line.split("\t")[1])
        spanish_sentence1.append(line.split("\t")[0])
        id_.append(str(index))
        qid1.append(str(index*2 + 1))
        qid2.append(str(index*2 + 2))
        index += 1
        line = f.readline()
    data_frame = pd.DataFrame({'id':id_,'qid1':qid1,'qid2':qid2,'english_sentence1':english_sentence1, 'spanish_sentence1':spanish_sentence1})
    data_frame.to_csv(save_pt,index=False,encoding='UTF-8')


if __name__ == '__main__':
    config_fp = 'C:/Users/jieyang/Desktop/GIT_code/Are-you-happy/conf/featwheel.conf'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    original_english_pt = '%s/%s' % (config.get('DIRECTORY', 'origin_spanish_pt'),config.get('FILE_NAME', 'cikm_english_train_20180516_txt'))
    original_spanish_pt = '%s/%s' % (config.get('DIRECTORY', 'origin_spanish_pt'),config.get('FILE_NAME', 'cikm_spanish_train_20180516_txt'))
    save_pt = '%s/%s' % (config.get('DIRECTORY', 'csv_spanish_pt'),config.get('FILE_NAME', 'cikm_english_and_spanish_train_20180516_noid_csv'))
    txtToCsvTrainEnglishandSpanishVersion(original_english_pt, original_spanish_pt, save_pt)