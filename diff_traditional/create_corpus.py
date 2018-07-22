from nltk.corpus import stopwords
import string
import re

filter_char = string.punctuation
digit_char = '123456789'
es_char = u'¿¡'
stops = set(stopwords.words('spanish'))

def filterChars(word):
	word = word.lower()
	word = word.strip(filter_char).strip(digit_char).strip(es_char)
	return word if word not in stops else ''

f_out = open('preprocess_data/corpus.txt','w')
# 生成西班语料库，方便之后处理tfidf等统计信息，防止overfitting
with open('source_data/cikm_english_train_20180516.txt','r') as f_in1, open(
	'source_data/cikm_spanish_train_20180516.txt','r') as f_in2,open(
	'source_data/cikm_test_a_20180516.txt','r') as f_in3,open(
	'source_data/cikm_unlabel_spanish_train_20180516.txt','r') as f_in4:
	for raw_line in f_in1:
		line = raw_line.strip('\n').split('\t')
		sent1 = ''
		for word in line[1].split(' '):
			word = filterChars(word)
			if word:
				sent1 += word + ' '
                
		sent2 = ''
		for word in line[3].split(' '):
			word = filterChars(word)
			if word:
				sent2 += word + ' '
                
		f_out.write(sent1 + '\n' + sent2 + '\n')

	for raw_line in f_in2:
		line = raw_line.strip('\n').split('\t')
		sent1 = ''
		for word in line[0].split(' '):
			word = filterChars(word)
			if word:
				sent1 += word + ' '
                
		sent2 = ''
		for word in line[2].split(' '):
			word = filterChars(word)
			if word:
				sent2 += word + ' '
		f_out.write(sent1 + '\n' + sent2 + '\n')

	for raw_line in f_in3:
		line = raw_line.strip('\n').split('\t')
		sent1 = ''
		for word in line[0].split(' '):
			word = filterChars(word)
			if word:
				sent1 += word + ' '
                
		sent2 = ''
		for word in line[1].split(' '):
			word = filterChars(word)
			if word:
				sent2 += word + ' '
		f_out.write(sent1 + '\n' + sent2 + '\n')

	for raw_line in f_in4:
		line = raw_line.strip('\n').split('\t')
		sent1 = ''
		for word in line[0].split(' '):
			word = filterChars(word)
			if word:
				sent1 += word + ' '
                
		f_out.write(sent1 + '\n')

f_out.close()

