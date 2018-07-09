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
		doc1 = nlp(q1)
		doc2 = nlp(q2)


nlp = spacy.load("en")
# nlp = spacy.load("es")

IndexedWord = namedtuple('IndexedWord','word index')
DependencyArc = namedtuple('DependencyArc','start_word end_word label')
doc = nlp(u"it is word tokenize test for spacy")

arcs = set()

for token in doc:
	newArc = DependencyArc(IndexedWord(token.head.text,token.head.i+1),IndexedWord(token.text,token.i+1),token.dep_)
	arcs.add(newArc)
	print(newArc)
	print()


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