# 建立单词字典，生成如{word ：idf_value} 的形式并保存
# 根据提供的word2vec生成针对此题目单词集的word2vec

import math
import string

words_dic = {}
total_sum = 0
print('Start sentences pair process')
with open('preprocess_data/sentences_pair.txt','r') as f_in:
	for raw_line in f_in:
		line = raw_line.strip('\n').split('\t')
		sentences = line[1].strip().split('\001')

		for sent in sentences:
			words = sent.split()
			for word in words:
				word = word.strip('¿¡').strip('12345678').strip(string.punctuation).lower()
				if word not in words_dic:
					words_dic[word] = 0
				words_dic[word] += 1
			total_sum += len(words)

	print('There are total {0} words'.format(total_sum))
print('Finish sentences pair process...')

print('Start words process...')
with open('preprocess_data/words.txt','w') as f_out:
	for key, value in words_dic.items():
		f_out.write(key + ' ' + str(math.log(total_sum/(float(value) + 1.0))) + '\n')		
print('Finish words process...')

print('Start word2vec process...')
with open("source_data/wiki.es.vec") as f_in, open('preprocess_data/word2vec.dict','w') as f_out:
	for raw_line in f_in:
		line = raw_line.strip('\n').split()
		if line[0] in words_dic:
			f_out.write(raw_line)
print('Finish word2vec process...')
