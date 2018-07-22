import pandas as pd
import re
from nltk.stem import SnowballStemmer
import pickle

train_e2s = '../diff_traditional/source_data/cikm_english_train_20180516.txt'
train_s2e = '../diff_traditional/source_data/cikm_spanish_train_20180516.txt'
testxt = '../diff_traditional/source_data/cikm_test_a_20180516.txt'

_stemmer = SnowballStemmer('spanish')

# remove punctuation
def removePunctuation(row):
	r = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~¿¡]+'
	line = re.sub(r,'',row)
	return line

# stemming
def stemming(row):
	line = [_stemmer.stem(word) for word in row.split()]
	return ' '.join(line)

with open(train_e2s,'r') as txtf1, open(train_s2e,'r') as txtf2, open(testxt,'r') as testf:
	train_e_s = pd.read_csv(txtf1,sep = '\t',header = None)
	train_s_e = pd.read_csv(txtf2,sep = '\t',header = None)
	testcsv = pd.read_csv(testf,sep = '\t',header = None)

	with open('preprocessing_train.csv','w') as trainff, open('preprocessing_test.csv','w') as testff:
		train_e_s.columns = ['eng_qura1', 'spa_qura1', 'eng_qura2', 'spa_qura2', 'label']
		train_s_e.columns = ['spa_qura1', 'eng_qura1', 'spa_qura2', 'eng_qura2', 'label']
		testcsv.columns = ['spa_qura1', 'spa_qura2']

		testcsv['spa_qura1'] = testcsv['spa_qura1'].apply(removePunctuation)
		#testcsv['spa_qura1'] = testcsv['spa_qura1'].apply(stemming)

		testcsv['spa_qura2'] = testcsv['spa_qura2'].apply(removePunctuation)
		#testcsv['spa_qura2'] = testcsv['spa_qura2'].apply(stemming)
		
		testcsv.to_csv(testff,index=False)
		print('haha')
		print('Finish testset preprocessing.....')
		print("begin")

		train = pd.concat([train_e_s,train_s_e]).drop(['eng_qura1','eng_qura2'],axis=1)

		train['spa_qura1'] = train['spa_qura1'].apply(removePunctuation)
		#train['spa_qura1'] = train['spa_qura1'].apply(stemming)

		train['spa_qura2'] = train['spa_qura2'].apply(removePunctuation)
		#train['spa_qura2'] = train['spa_qura2'].apply(stemming)
		
		train.to_csv(trainff,index=False)
		print('Finish train set preprocessing.....')

# with open(testxt,'r') as testf:
# 	testcsv = pd.read_csv(testf,sep = '\t',header = None)

# 	with open('preprocessing_test.csv','w') as ff:
# 		testcsv.columns = ['spa_qura1', 'spa_qura2']

# 		testcsv['spa_qura1'] = testcsv['spa_qura1'].apply(removePunctuation)
# 		#testcsv['spa_qura1'] = testcsv['spa_qura1'].apply(stemming)

# 		testcsv['spa_qura2'] = testcsv['spa_qura2'].apply(removePunctuation)
# 		#testcsv['spa_qura2'] = testcsv['spa_qura2'].apply(stemming)
		
# 		testcsv.to_csv(ff,index=False)
# 		print('Finish test set preprocessing.....')






