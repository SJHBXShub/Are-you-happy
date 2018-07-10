# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 20:39
# @Author  : Ruiko
# @Email   :


import spacy
from collections import namedtuple
from extractor import Extractor
# if there is an error "no module named en"
# use "python -m spacy download en" in terminal
class SpacySVOExtract(Extractor):
	def __init__(self, config_fp):
		Extractor.__init__(self, config_fp)

	def extract_row(self, row):
		q1 = str(row['spanish_sentence1'])
		q2 = str(row['spanish_sentence2'])

		nlp = spacy.load("en")

		doc1 = nlp(q1)
		doc2 = nlp(q2)

		feature_SVO = self.extract_SVO(doc1)
		return feature_SVO

	def extract_SVO(self, doc):

		IndexedWord = namedtuple('IndexedWord', 'word index')
		DependencyArc = namedtuple('DependencyArc', 'start_word end_word label')

		arcs = set()
		fs = list()

		for token in doc:
			newArc = DependencyArc(IndexedWord(token.head.text, token.head.i + 1), IndexedWord(token.text, token.i + 1),
								   token.dep_)
			arcs.add(token)
			print(token)
			print(token.dep_)
			if token.dep_ == "nsubj" or token.dep_ == "ROOT" or token.dep_ == "attr":
				temp = list()
				temp.append(token)
				temp.append(token.dep_)
				fs.append(temp)
				print(fs)

			return fs




# nlp = spacy.load("en")
# # nlp = spacy.load("es")
#
# IndexedWord = namedtuple('IndexedWord','word index')
# DependencyArc = namedtuple('DependencyArc','start_word end_word label')
# doc = nlp(u"it is word tokenize test for spacy")
#
# arcs = set()
# fs = list()
#
# for token in doc:
# 	newArc = DependencyArc(IndexedWord(token.head.text,token.head.i+1),IndexedWord(token.text,token.i+1),token.dep_)
# 	arcs.add(token)
# 	print(token)
# 	print(token.dep_)
# 	if token.dep_ == "nsubj" or token.dep_ == "ROOT" or token.dep_ == "attr":
# 		temp = list()
# 		temp.append(token)
# 		temp.append(token.dep_)
# 		fs.append(temp)
# 		print(fs)

	# print(arcs)


# print(doc)
# for d in doc:
#     print(d)
# test_doc = nlp(u"you are best. it is lemmatize test for spacy. I love these books")
# for t in test_doc:
#     print(t , t.lemma_ , t.lemma) #词干化
# for t in test_doc:
#     print(t , t.pos_ , t.pos) #词性标注
# test_doc = nlp(u"Rami Eid is studying at Stony Brook University in New York")
# for ent in test_doc.ents:
#     print(ent, ent.label_, ent.label)