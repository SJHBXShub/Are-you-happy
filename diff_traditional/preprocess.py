# 去除西班牙文本中的标点符号，数字符号，以及停用词，得到的语句对并单独保存
import string
import re
from nltk.corpus import stopwords

filter_char = string.punctuation
digit_char = '123456789'
es_char = u'¿¡'
stops = set(stopwords.words('spanish'))

def filterChars(word):
	word = word.lower()
	word = word.strip(filter_char).strip(digit_char).strip(es_char)
	return word

with open('source_data/cikm_english_train_20180516.txt','r') as f_in1, open(
	'source_data/cikm_spanish_train_20180516.txt','r') as f_in2, open(
	'source_data/cikm_test_a_20180516.txt','r') as f_in3, open(
	'preprocess_data/sentences_pair.txt','w') as f_out:
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

		if len(sent1) == 0 or len(sent2) == 0:
			continue

		# 生成 lable sent1 sent2的形式
		f_out.write(line[-1] + '\t' + sent1.strip() + '\001' + sent2.strip() + '\n')

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

		if len(sent1) == 0 or len(sent2) == 0:
			continue

		# 生成 lable sent1 sent2的形式
		f_out.write(line[-1] + '\t' + sent1.strip() + '\001' + sent2.strip() + '\n')

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

		# 生成 lable sent1 sent2的形式
		f_out.write('-1' + '\t' + sent1.strip() + '\001' + sent2.strip() + '\n')

