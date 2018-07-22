import runner
import configparser as ConfigParser
import pandas as pd
from utils import MyUtil


def demo():
    '''
    MyUtil.getPredictFeature('train')
    MyUtil.getPredictFeature('test')
    '''
    config_fp = '../conf/featwheel.conf'
    #cv_exrc = runner.CrossValidation(config_fp)
    cv_exrc = runner.SingleExec(config_fp)
    cv_exrc.run_offline()
    cv_exrc.run_online()
    
    print("I am ok")
if __name__ == '__main__':
    demo()