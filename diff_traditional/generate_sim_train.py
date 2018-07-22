import pandas as pandas
import numpy as np
import gensim
import sys

doc2vec_size = 300

def read_corpus(filename):
	sent = []
	for raw_line in open(filename):
		line = raw_line.strip('\n')
		if line:
			sent.append(line)
	for index, line in enumerate(sent):
		yield gensim.models.doc2vec.TaggedDocument(line.split(),['%s' % index])

def doc2vec_train(filename):
	train_corpus = list(read_corpus(filename))
	model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_size, min_count=1, epochs=50, workers=7)
	model.build_vocab(train_corpus)
	model.train(train_corpus, total_examples = model.corpus_count, epochs = model.iter)
	model.save('doc2vec_model.dat')

filename = 'preprocess_data/corpus.txt'
doc2vec_train(filename)
print('Doc2Vec train finished ...')


