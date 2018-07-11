import runner
import configparser as ConfigParser
import pandas as pd
from utils import LogUtil
from feature import Feature

def getShFeature():
    ShFeature_pt = './sentence_sim_feature.dat'
    data_feature_fp = './sentence_representation_distance.preprocessing_train_merge.csv.smat'
    data_feature_fp1 = './sentence_representation_tfidf_distance.preprocessing_train_merge.csv.smat'
    feature_file = open(data_feature_fp, 'w')
    feature_file1 = open(data_feature_fp1, 'w')
    sh_feature = open(ShFeature_pt,'r')
    feature_file.write('%d %d\n' % (21400, 1))
    feature_file1.write('%d %d\n' % (21400, 1))
    line = sh_feature.readline()
    index = 0

    while line:
    	index += 1
    	if index <= 21400:
    	    feature1 = [line.split(',')[0]]
    	    feature2 = [line.split(',')[1][0:len(line.split(',')[1])-1]]
    	    Feature.save_feature(feature1, feature_file)
    	    Feature.save_feature(feature2, feature_file1)
    	line = sh_feature.readline()
'''
feature_file = open(self.data_feature_fp, 'w')
Feature.save_feature(feature, feature_file)
'''



def demo():
    config_fp = '../conf/featwheel.conf'
    #getShFeature()
    
    cv_exrc = runner.SingleExec(config_fp)
    cv_exrc.run_offline()
    cv_exrc.run_online()
    print("I am ok")
if __name__ == '__main__':
    demo()