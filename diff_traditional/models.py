from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn import metrics
import numpy as np
import numpy
basic_feature = ['len_word_s1', 'len_word_s2', 'len_char_s2', 'len_char_s1', 'len_ratio']
fuzz_feature = ['fuzz_QRatio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
gramoverlap_feature = ['1-gramoverlap_word', '2-gramoverlap_word', '3-gramoverlap_word', '2-gramoverlap_char', '3-gramoverlap_char', '4-gramoverlap_char', '5-gramoverlap_char']
other_feature = ['sentence_distance', 'sentence_distance_tfidf', 'lcs_diff', 'has_no_word', 'sentence_similarity']
model_feature = ['cnn_result']#,'matchPyramid_result']
word_feature = ['word_match_share']
sequence_feature = ['long_common_sequence', 'long_common_substring', 'long_common_suffix', 'long_common_prefix']
#, 'levenshtein_distance']
word2vec_feature_ave_idf = ['euclidean_distance_ave_idf', 'cosine_distance_ave_idf', 'canberra_distance_ave_idf', 'jaccard_distance_ave_idf', 'minkowski_distance_ave_idf', 'skew_s1vec_ave_idf', 'skew_s2vec_ave_idf', 'kur_s1vec_ave_idf', 'kur_s2vec_ave_idf', 'kendalltau_coff_ave_idf']

feature = []
feature.extend(basic_feature)
feature.extend(fuzz_feature) 
feature.extend(gramoverlap_feature) 
feature.extend(sequence_feature) 
feature.extend(other_feature) 
feature.extend(model_feature)
feature.extend(word_feature)
feature.extend(word2vec_feature_ave_idf)

print ('%s features' %(len(feature)))
def shuffle(X, y):
	m = X.shape[0]
	ind = np.arange(m)
	for i in range(7):
		np.random.shuffle(ind)
	return X[ind], y[ind]

data = pd.read_csv('train2.dat')
data.fillna(0, inplace=True)

X_train = data[data.label >= 0][feature].values
y_train = data[data.label >= 0]['label'].values

X_test = data[feature][data.label == -1].values

X_train, y_train = shuffle(X_train,y_train)


gbdt_model = GradientBoostingClassifier(n_estimators=300, max_depth=6, loss="deviance")
logloss = cross_val_score(gbdt_model, X_train, y_train, cv=7, scoring='neg_log_loss')
print ('logloss: ', -logloss.mean())

gbdt_model.fit(X_train, y_train)
pred = gbdt_model.predict_proba(X_test)[:,1]
np.savetxt("result_gbdt.txt", pred, fmt='%1.6f')
print("mean: ",np.mean(pred))


