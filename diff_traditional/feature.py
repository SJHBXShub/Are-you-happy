import math
import numpy as np
import pandas as pd
import scipy
import gensim

from scipy.stats import skew, kurtosis
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel
from nltk.corpus import stopwords

words_dict = {}
with open('preprocess_data/words.txt','r') as f_in:
	for raw_line in f_in:
		line = raw_line.strip('\n').split()
		if len(line) != 2:
			continue
		words_dict[line[0]] = float(line[1])

word2vec_dict = {}
with open('source_data/wiki.es.vec','r') as f_in:
	for raw_line in f_in:
		line = raw_line.strip('\n\r').split()
		if line[0] in words_dict and line[1] not in words_dict:
			try:
				word2vec_dict[line[0]] = [float(i) for i in line[1:]]
			except:
				continue
			#word2vec_dict[line[0]] = [float(i) for i in line[1:]]

stops = set(stopwords.words('spanish'))
doc2vec_model = gensim.models.Doc2Vec.load('doc2vec_model.dat')
cnn_result = pd.read_csv('source_data/cnn2result.dat')
#matchPyramid_result = pd.read_csv('source_data/matchPyramid_result.dat')

def read_data(filename):
	return pd.read_csv(filename, names=['label', 'sentences'], sep='\t')

# 欧式距离
def euclidean(vec1, vec2):
	num = len(vec1)
	s = 0.0
	for i in range(num):
		s += (vec1[i] - vec2[i])**2
	return math.sqrt(s)

# 余弦距离
def cos(vec1, vec2):
	length = len(vec1)
	num = 0.0
	vec1_norm = 0.0
	vec2_norm = 0.0
	for i in range(length):
		num += vec1[i] * vec2[i]
		vec1_norm += vec1[i] * vec1[i]
		vec2_norm += vec2[i] * vec2[i]
	norm = math.sqrt(vec1_norm) * math.sqrt(vec2_norm)
	if norm == 0:
		return 0
	return num / norm

# word feature
def word_match_share(sentences):
	sent1_words = {}
	sent2_words = {}

	for word in sentences.split("\001")[0].split():
		if word not in stops:
			sent1_words[word] = sent1_words.get(word, 0) + 1

	for word in sentences.split("\001")[1].split():
		if word not in stops:
			sent2_words[word] = sent2_words.get(word, 0) + 1

	shared_word_in_sen1 = sum([sent1_words[w] for w in sent1_words if w in sent2_words])
	sheare_word_in_sen2 = sum([sent1_words[w] for w in sent1_words if w in sent2_words])
	total = sum(sent1_words.values()) + sum(sent2_words.values())

	if total <= 0:
		return 0.
	else:
		return (shared_word_in_sen1 + sheare_word_in_sen2) / total

# update feature
# 根据语料库训练出的doc2vec model 来得到句向量
def sim_by_Doc2vec(model,sentences):
	sent_vec = []
	for sen in sentences.split("\001"):
		M = []
		for word in sen.split():
			try:
				M.append((model.wv[word]))
			except :
				continue
		M = np.array(M)
		v = M.sum(axis = 0)
		norm = np.sqrt((v**2).sum())
		sent_vec.append(v / norm)
	return cos(sent_vec[0], sent_vec[1])

# 计算两个句子间的余弦距离
def sentence_distance(sentences):
	words = {}
	pos = 0
	for sen in sentences.split("\001"):
		for word in sen.split():
			if word not in words:
				words[word] = pos
				pos += 1
	sen_vec_1 = [ 0 for i in range(pos) ]
	sen_vec_2 = [ 0 for i in range(pos) ]
	sens = sentences.split("\001")
	for i, sen in enumerate(sens):
		for word in sen.split():
			if i == 0:
				sen_vec_1[words[word]] = 1
			else:
				sen_vec_2[words[word]] = 1
	return cos(sen_vec_1, sen_vec_2)

# 计算两个句子间的余弦距离（算上了TFIDF的值）
def sentence_distance_tfidf(sentences):
	words = {}
	pos = 0
	for sen in sentences.split("\001"):
		for word in sen.split():
			if word not in words:
				words[word] = pos
				pos += 1
	sen_vec_1 = [ 0 for i in range(pos) ]
	sen_vec_2 = [ 0 for i in range(pos) ]
	sens = sentences.split("\001")
	for i, sen in enumerate(sens):
		for word in sen.split():
			try:
				if i == 0:
					sen_vec_1[words[word]] = words_dict[word]
				else:
					sen_vec_2[words[word]] = words_dict[word]
			except:
				if i == 0:
					sen_vec_1[words[word]] = 0
				else:
					sen_vec_2[words[word]] = 0
	return cos(sen_vec_1, sen_vec_2)

# 公共子序列的位置差
def lcs_diff(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)

	dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
	offset = 0
	for i in range(1,len1):
		for j in range(1,len2):
			if sen1[i] == sen2[j]:
				dp[i][j] = dp[i-1][j-1] + 1
				of = abs(j - i)
				offset += of
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	return offset * 1.0 / (len1 + len2)

# 最长公共子序列
def long_common_sequence(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)

	dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
	for i in range(len1):
		for j in range(len2):
			if sen1[i] == sen2[j]:
				dp[i][j] = dp[i-1][j-1] + 1
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	return dp[len1-1][len2-1] * 1.0 / (len1 + len2)

def long_common_prefix(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)

	max_prefix = 0
	min_len = min(len1, len2)
	for i in range(min_len):
		if sen1[i] == sen2[i]:
			max_prefix += 1
	return max_prefix * 1.0 / (len1 + len2)

def long_common_suffix(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)
	sen1.reverse()
	sen2.reverse()
	min_len = min(len1, len2)
	max_suffix = 0
	for i in range(min_len):
		if sen1[i] == sen2[i]:
			max_suffix += 1
	return max_suffix * 1.0 / (len1 + len2)

def long_common_substring(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)

	dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
	for i in range(len1):
		for j in range(len2):
			if sen1[i] == sen2[j]:
				dp[i][j] = dp[i-1][j-1] + 1
			else:
				dp[i][j] = 0
	return dp[len1-1][len2-1] * 1.0 / (len1 + len2)

def levenshtein_distance(sentences):
	sen = sentences.split("\001")
	s = sen[0].split(" ")
	t = sen[1].split(" ")

	if s == t: return 0
	elif len(s) == 0: return len(t)
	elif len(t) == 0: return len(s)
	v0 = [None] * (len(t) + 1)
	v1 = [None] * (len(t) + 1)
	for i in range(len(v0)):
		v0[i] = i
	for i in range(len(s)):
		v1[0] = i + 1
		for j in range(len(t)):
			cost = 0 if s[i] == t[j] else 1
			v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)  
		for j in range(len(v0)):
			v0[j] = v1[j]  
	return v1[len(t)] / (len(s) + len(t))

def n_gram_over_lap(sen1, sen2, n):
	len1 = len(sen1)
	len2 = len(sen2)

	word_set1 = set()
	word_set2 = set()

	for i in range(len1):
		if i <= n-2:
			continue
		join_str = ""
		for j in range(n-1, -1, -1):
			join_str += sen1[i-j]
		word_set1.add(join_str)

	for i in range(len2):
		if i <= n-2:
			continue
		join_str = ""
		for j in range(n-1, -1, -1):
			join_str += sen2[i-j]
		word_set2.add(join_str)
	num1 = len(word_set1 & word_set2)
	num2 = len(word_set1) + len(word_set2)
	if num2 == 0:
		return 0
	return num1 * 1.0 / num2

def n_gram_over_lap_char(sentences, n):
	sen = sentences.split("\001")
	sen1 = ''.join(sen[0].split(" "))
	sen2 = ''.join(sen[1].split(" "))

	return n_gram_over_lap(sen1, sen2, n)

def n_gram_over_lap_word(sentences, n):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")

	return n_gram_over_lap(sen1, sen2, n)

def len_ratio(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")
	len1 = len(sen1)
	len2 = len(sen2)
	if len2 == 0:
	    len2 == 1
	return len1 * 1.0 / len2

def has_no_word(sentences):
	sen = sentences.split("\001")
	sen1 = sen[0].split(" ")
	sen2 = sen[1].split(" ")

	if "no" in sen1 and "no" in sen2:
		return 1
	if "no" not in sen1 and "no" not in sen2:
		return 1
	if "no" not in sen1 and "no" in sen2:
		return 0
	if "no" in sen1 and "no" not in sen2:
		return 0

def fuzz_QRatio(sentences):
	sen = sentences.split("\001")
	return fuzz.QRatio(sen[0], sen[1]) 

def fuzz_WRatio(sentences):
	sen = sentences.split("\001")
	return fuzz.WRatio(sen[0], sen[1])

def fuzz_partial_ratio(sentences):
	sen = sentences.split("\001")
	return fuzz.partial_ratio(sen[0], sen[1])

def fuzz_partial_token_set_ratio(sentences):
	sen = sentences.split("\001")
	return fuzz.partial_token_set_ratio(sen[0], sen[1])

def fuzz_partial_token_sort_ratio(sentences):
	sen = sentences.split("\001")
	return fuzz.partial_token_sort_ratio(sen[0], sen[1])

def fuzz_token_set_ratio(sentences):
	sen = sentences.split("\001")
	return fuzz.token_set_ratio(sen[0], sen[1])

def fuzz_token_sort_ratio(sentences):
	sen = sentences.split("\001")
	return fuzz.token_sort_ratio(sen[0], sen[1])

def sent2vec(sen):
	M = [[ 0 for i in range(300) ]]
	words = sen.split()
	for w in words:
		try:
			M.append(word2vec_dict[w])
		except:
			continue
	M = np.array(M)
	v = M.sum(axis = 0)
	norm = np.sqrt((v**2).sum())
	return v / norm

def sent2vec_ave(sen):
	M = [[ 0 for i in range(300) ]]
	words = sen.split()
	for w in words:
		try:
			M.append(word2vec_dict[w])
		except:
			continue
	M = np.array(M)
	num = M.shape[0]
	v = M.sum(axis=0)
	return v / num

def sent2vec_ave_idf(sen):
	M = [[ 0 for i in range(300) ]]
	words = sen.split()
	for w in words:
		try:
			M.append([ words_dict[w] * x for x in word2vec_dict[w] ])
		except:
			continue
	M = np.array(M)
	num = M.shape[0]
	v = M.sum(axis=0)
	return v / num

def generate_feature(data):

	print ('basic feature ...')
	#length of sentence
	data['len_word_s1'] = data.apply(lambda x: len(x['sentences'].split("\001")[0].split(" ")), axis=1)
	data['len_word_s2'] = data.apply(lambda x: len(x['sentences'].split("\001")[1].split(" ")), axis=1)
	data['len_ratio'] = data.apply(lambda x: len_ratio(x['sentences']), axis=1)
	data['len_char_s1'] = data.apply(lambda x: len(''.join(x['sentences'].split("\001")[0].split(" "))), axis=1)
	data['len_char_s2'] = data.apply(lambda x: len(''.join(x['sentences'].split("\001")[1].split(" "))), axis=1)

	print ('fuzzywuzzy feature ...')
	data['fuzz_QRatio'] = data.apply(lambda x: fuzz_QRatio(x['sentences']), axis=1)
	data['fuzz_WRatio'] = data.apply(lambda x: fuzz_WRatio(x['sentences']), axis=1)
	data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz_partial_ratio(x['sentences']), axis=1)
	data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz_partial_token_set_ratio(x['sentences']), axis=1)
	data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz_partial_token_sort_ratio(x['sentences']), axis=1)
	data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz_token_set_ratio(x['sentences']), axis=1)
	data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz_token_sort_ratio(x['sentences']), axis=1)

	print ('sequence feature ...')
	data['long_common_sequence'] = data.apply(lambda x: long_common_sequence(x['sentences']), axis=1)
	data['long_common_prefix'] = data.apply(lambda x: long_common_prefix(x['sentences']), axis=1)
	data['long_common_suffix'] = data.apply(lambda x: long_common_suffix(x['sentences']), axis=1)
	data['long_common_substring'] = data.apply(lambda x: long_common_substring(x['sentences']), axis=1)
	data['levenshtein_distance'] = data.apply(lambda x: levenshtein_distance(x['sentences']), axis=1)
    
	print ('word feature...')
	data['word_match_share'] = data.apply(lambda x: word_match_share(x['sentences']), axis=1)

	print ('other feature and n-gram feature ...')
	data['sentence_similarity'] = data.apply(lambda x: sim_by_Doc2vec(doc2vec_model ,x['sentences']), axis=1)
	data['has_no_word'] = data.apply(lambda x: has_no_word(x['sentences']), axis=1)
	data['sentence_distance'] = data.apply(lambda x: sentence_distance(x['sentences']), axis=1)
	data['sentence_distance_tfidf'] = data.apply(lambda x: sentence_distance_tfidf(x['sentences']), axis=1)
	data['lcs_diff'] = data.apply(lambda x: lcs_diff(x['sentences']), axis=1)
	data['cnn_result'] = cnn_result['0']
	#data['matchPyramid_result'] = matchPyramid_result['0']

	#N-gramOverlap
	data['1-gramoverlap_word'] = data.apply(lambda x: n_gram_over_lap_word(x['sentences'], 1), axis=1)
	data['2-gramoverlap_word'] = data.apply(lambda x: n_gram_over_lap_word(x['sentences'], 2), axis=1)
	data['3-gramoverlap_word'] = data.apply(lambda x: n_gram_over_lap_word(x['sentences'], 3), axis=1)
	data['2-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 2), axis=1)
	data['3-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 3), axis=1)
	data['4-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 4), axis=1)
	data['5-gramoverlap_char'] = data.apply(lambda x: n_gram_over_lap_char(x['sentences'], 5), axis=1)

	print ('sen2vec feature ...')
	sent1_vectors = np.zeros((data.shape[0], 300))
	for i, sents in enumerate(data.sentences.values):
		sent = sents.split('\001')[0]
		sent1_vectors[i, :] = sent2vec_ave_idf(sent)

	sent2_vectors = np.zeros((data.shape[0], 300))
	for i, sents in enumerate(data.sentences.values):
		sent = sents.split('\001')[1]
		sent2_vectors[i, :] = sent2vec_ave_idf(sent)

	# data['cityblock_distance_ave_idf'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['jaccard_distance_ave_idf'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['cosine_distance_ave_idf'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['canberra_distance_ave_idf'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['euclidean_distance_ave_idf'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['braycurtis_distance_ave_idf'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['minkowski_distance_ave_idf'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['pearson_coff_ave_idf'] = [scipy.stats.pearsonr(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['spearman_coff_ave_idf'] = [scipy.stats.spearmanr(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['kendalltau_coff_ave_idf'] = [scipy.stats.kendalltau(x, y)[0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]

	# data['polynomial_kernel_ave_idf'] = [polynomial_kernel(x.reshape(1, -1), y.reshape(1, -1))[0][0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	# data['sigmoid_kernel_ave_idf'] = [sigmoid_kernel(x.reshape(-1, 1), y.reshape(-1, 1))[0][0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	# data['rbf_kernel_ave_idf'] = [rbf_kernel(x.reshape(-1, 1), y.reshape(-1, 1))[0][0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	# data['laplacian_kernel_ave'] = [laplacian_kernel(x.reshape(-1, 1), y.reshape(-1, 1))[0][0] for (x, y) in zip(np.nan_to_num(sent1_vectors), np.nan_to_num(sent2_vectors))]
	data['skew_s1vec_ave_idf'] = [skew(x) for x in (sent1_vectors)]
	data['skew_s2vec_ave_idf'] = [skew(x) for x in (sent2_vectors)]
	data['kur_s1vec_ave_idf'] = [kurtosis(x) for x in (sent1_vectors)]
	data['kur_s2vec_ave_idf'] = [kurtosis(x) for x in (sent2_vectors)]

	return data

if __name__ == '__main__':
	print('Begin feature process')

	filename = 'preprocess_data/sentences_pair.txt'
	data = read_data(filename)
	train_data = generate_feature(data)
	train_data.to_csv('train2.dat',index = False)



