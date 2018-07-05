from utils import DataUtil
import configparser as ConfigParser
import pandas as pd

def saveLabel(data_set_name):
    config_fp = '../conf/featwheel.conf'
    config = ConfigParser.ConfigParser()
    config.read(config_fp)
    data = pd.read_csv('%s/%s' % (config.get('DIRECTORY', 'csv_spanish_cleaning_pt'), data_set_name)).fillna(value="")
    labels_pt = '%s/%s.label' % (config.get('DIRECTORY', 'label_pt'),config.get('FEATURE', 'offline_rawset_name'))

    labels = []
    for index, row in data.iterrows():
        cur_label = str(row['is_duplicate'])
        labels.append(cur_label)

    DataUtil.save_vector(labels_pt,labels,'w')

if __name__ == '__main__':
    precess_file_name = 'preprocessing_train_merge.csv'
    saveLabel(precess_file_name)



