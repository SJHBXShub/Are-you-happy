# 生成powerful word 单词表，计算数据中词语影响力，格式如下
# 词语 --> [0. 出现在语句对的数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. # 双侧语句对正确比例]
# 在生成powerful word表后， 我们可以根据阀值选出合适的单词来构造feature

import pandas as pd
import numpy as np

def generate_powerful_word(filename):
	"""
	input: sentence pair file
	output: word dict
	"""
	data = pd.read_csv(filename, names=['label', 'sentences'], sep='\t')
	power_words = {}
	for index, row in data.iterrows():
		label = int(row['label'])
		sen1_words = row['sentences'].split("\001")[0].split()
		sen2_words = row['sentences'].split("\001")[1].split()
		all_words = set(sen1_words + sen2_words)
		sen1_words = set(sen1_words)
		sen2_words = set(sen2_words)
		for w in all_words:
			if w not in power_words:
				power_words[w] = [0. for i in range(7)]
			# 计算word出现的语句对数量
			power_words[w][0] += 1.
			power_words[w][1] += 1.

			# 计算单侧语句数量
			if (w in sen1_words and w not in sen2_words) or (w not in sen1_words and w in sen2_words):
				power_words[w][3] += 1.
				if label == 1:
					power_words[w][2] += 1.
					power_words[w][4] += 1.

			# 计算双侧语句数量
			if w in sen1_words and w in sen2_words:
				power_words[w][5] += 1.
				if label == 1:
					power_words[w][2] += 1.
					power_words[w][6] += 1.
	for w in power_words:
		power_words[w][1] /= len(data)
		# 正确语句对比例
		power_words[w][2] /= power_words[w][0]
		if power_words[w][3] > 0:
			# 单侧正确比例
			power_words[w][4] /= power_words[w][3]
		if power_words[w][5] > 0:
			#双侧正确比例
			power_words[w][6] /= power_words[w][5]

		#单侧语句比例
		power_words[w][3] /= power_words[w][0]
		#双侧语句比例
		power_words[w][5] /= power_words[w][0]

	sort_power_word = sorted(power_words.items(), key = lambda x:x[1][0], reverse = True)
	print('There are total %d power words' % len(sort_power_word))
	return sort_power_word

def save_powerful_word(power_words, filename):
	f = open(filename, 'w')
	for ele in power_words:
		f.write("%s" % ele[0])
		for num in ele[1]:
			f.write("\t%.5f" % num)
		f.write("\n")
	f.close()

with open('preprocess_data/sentences_pair.txt') as f_in:
	power_words = generate_powerful_word(f_in)
	save_powerful_word(power_words,'preprocess_data/power_words.txt')




