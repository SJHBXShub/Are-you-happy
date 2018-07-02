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

stops = set(stopwords.words("english"))
snowball_stemmer = SnowballStemmer('english')

class Length(Extractor):
    def extract_row(self, row):
        q1 = str(row['english_sentence1'])
        q2 = str(row['english_sentence2'])

        fs = list()
        fs.append(len(q1))
        fs.append(len(q2))
        fs.append(len(q1.split()))
        fs.append(len(q2.split()))
        return fs

    def get_feature_num(self):
        return 4


def demo():
    #Need_change
    config_fp = 'C:/Users/jieyang/Desktop/GIT_code/Are-you-happy/conf/featwheel.conf'
    Length(config_fp).extract('train')


if __name__ == '__main__':
    demo()
