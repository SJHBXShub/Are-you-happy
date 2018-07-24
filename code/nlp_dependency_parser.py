# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 20:39
# @Author  : Ruiko
# @Email   :


import spacy
import csv
from collections import namedtuple
from extractor import Extractor
# if there is an error "no module named en"
# use "python -m spacy download en" in terminal

# SVO Extract in English
class SpacySVOExtractEnglish1(Extractor):
    def __init__(self, config_fp, language = "en"):
        Extractor.__init__(self, config_fp)
        # should be modified
        self.language = language
    # should be modified
    
    def extract_row(self, row):
        # input sentences
        q1 = str(row['english_sentence1'])
        # extract English
        nlp = spacy.load("en")
        
        doc1 = nlp(q1)
        
        feature_SVO = self.extract_SVO(doc1)
        
        # return a list which contain Subject Verb Object
        return feature_SVO
    
    def extract_SVO(self, doc):
        # Extract function in low level
        IndexedWord = namedtuple('IndexedWord', 'word index')
        DependencyArc = namedtuple('DependencyArc', 'start_word end_word label')
        
        arcs = set()
        fs = list()
        
        list_temp = ["", "", ""]
        for token in doc:
            
            print(token)
            # nsubj -> subject, root -> verb or link verb, dobj -> object, attr -> attribute
            newArc = DependencyArc(IndexedWord(token.head.text, token.head.i + 1), IndexedWord(token.text, token.i + 1),
                                   token.dep_)
                                   arcs.add(token)
                                   
                                   if token.dep_ == "nsubj":
                                       print(token)
                                       list_temp[0] += (str(token) + " ")
                                   if token.dep_ == "ROOT":
                                       print(token)
                                       list_temp[1] += (str(token) + " ")
                                           if token.dep_ == "attr" or token.dep_ == "dobj":
                                               print(token)
                                               list_temp[2] += (str(token) + " ")
                                           fs.append(list_temp)
                                           out = open('English1_SVO_feature.csv', 'a', newline='')
                                               csv_write = csv.writer(out, dialect='excel')
                                                   csv_write.writerow(list_temp)
                                                   out.close()
                                                   print(fs)
                                                   # if token.dep_ == "nsubj" or token.dep_ == "ROOT" or token.dep_ == "attr" or token.dep_ == "dobj":
                                                   #     temp = list()
                                                   #     temp.append(token)
                                                   #     temp.append(token.dep_)
                                                   #     fs.append(temp)
                                                   #     print(fs)
                                                   #     put the sentence constituents into lists
                                                       return fs

def get_feature_num(self):
    return 1


class SpacySVOExtractEnglish2(Extractor):
    def __init__(self, config_fp, language = "en"):
        Extractor.__init__(self, config_fp)
        # should be modified
        self.language = language
    # should be modified
    
    def extract_row(self, row):
        # input sentences
        q1 = str(row['english_sentence2'])
        # extract English
        nlp = spacy.load("en")
        
        doc1 = nlp(q1)
        
        feature_SVO = self.extract_SVO(doc1)
        
        # return a list which contain Subject Verb Object
        return feature_SVO
    
    def extract_SVO(self, doc):
        # Extract function in low level
        IndexedWord = namedtuple('IndexedWord', 'word index')
        DependencyArc = namedtuple('DependencyArc', 'start_word end_word label')
        
        arcs = set()
        fs = list()
        
        list_temp = ["", "", ""]
        for token in doc:
            
            print(token)
            # nsubj -> subject, root -> verb or link verb, dobj -> object, attr -> attribute
            newArc = DependencyArc(IndexedWord(token.head.text, token.head.i + 1), IndexedWord(token.text, token.i + 1),
                                   token.dep_)
                                   arcs.add(token)
                                   
                                   if token.dep_ == "nsubj":
                                       print(token)
                                       list_temp[0] += (str(token) + " ")
                                   if token.dep_ == "ROOT":
                                       print(token)
                                       list_temp[1] += (str(token) + " ")
                                           if token.dep_ == "attr" or token.dep_ == "dobj":
                                               print(token)
                                               list_temp[2] += (str(token) + " ")
                                           fs.append(list_temp)
                                           out = open('English2_SVO_feature.csv', 'a', newline='')
                                               csv_write = csv.writer(out, dialect='excel')
                                                   csv_write.writerow(list_temp)
                                                   out.close()
                                                   print(fs)
                                                   # if token.dep_ == "nsubj" or token.dep_ == "ROOT" or token.dep_ == "attr" or token.dep_ == "dobj":
                                                   #     temp = list()
                                                   #     temp.append(token)
                                                   #     temp.append(token.dep_)
                                                   #     fs.append(temp)
                                                   #     print(fs)
                                                   #     put the sentence constituents into lists
                                                       return fs

def get_feature_num(self):
    return 1


class SpacySVOExtractSpanish1(Extractor):
    def __init__(self, config_fp, language="es"):
        Extractor.__init__(self, config_fp)
        self.language = language
    # same as above
    def extract_row(self, row):
        q1 = str(row['spanish_sentence1'])
        
        nlp = spacy.load("es")
        
        doc1 = nlp(q1)
        
        feature_SVO = self.extract_SVO(doc1)
        
        return feature_SVO
    
    def extract_SVO(self, doc):
        # same as above
        IndexedWord = namedtuple('IndexedWord', 'word index')
        DependencyArc = namedtuple('DependencyArc', 'start_word end_word label')
        
        arcs = set()
        fs = list()
        
        list_temp = ["", "", ""]
        for token in doc:
            
            print(token)
            print(token.dep_)
            print("============")
            # nsubj -> subject, root -> verb or link verb, dobj -> object, attr -> attribute
            newArc = DependencyArc(IndexedWord(token.head.text, token.head.i + 1), IndexedWord(token.text, token.i + 1),
                                   token.dep_)
                                   arcs.add(token)
                                   
                                   if token.dep_ == "nsubj":
                                       print(token)
                                       list_temp[0] += (str(token) + " ")
                                   if token.dep_ == "ROOT":
                                       print(token)
                                       list_temp[1] += (str(token) + " ")
                                           if token.dep_ == "obl" or token.dep_ == "obj":
                                               print(token)
                                               list_temp[2] += (str(token) + " ")
                                           fs.append(list_temp)
                                           out = open('Spanish1_SVO_feature.csv', 'a', newline='')
                                               csv_write = csv.writer(out, dialect='excel')
                                                   csv_write.writerow(list_temp)
                                                   out.close()
                                                   print(fs)
                                                   return fs


def get_feature_num(self):
    return 1




class SpacyNounExtractEnglish1(Extractor):
    def __init__(self, config_fp, language = "en"):
        Extractor.__init__(self, config_fp)
        # should be modified
        self.language = language
    # should be modified
    
    def extract_row(self, row):
        # input sentences
        q1 = str(row['english_sentence1'])
        # extract English
        nlp = spacy.load("en")
        
        doc1 = nlp(q1)
        
        feature_Noun = self.extract_Noun(doc1)
        
        # return a list which contain Subject Verb Object
        return feature_Noun
    
    def extract_Noun(self, doc):
        # Extract function in low level
        
        arcs = set()
        fs = list()
        
        all_tags = {w.pos: w.pos_ for w in doc}
        list_temp = ["", ""]
        # all tags of first sentence of our document
        for word in list(doc.sents)[0]:
            if word.tag_ == "NN" or word.tag_ == "NNS" or word.tag_ == "NNP" or word.tag_ == "NNPS":
                #NN，NNS，NNP，NNPS ： 名词，名词复数，专有名词，专有名词复数
                #VB，VBD，VBN，VBZ，VBP，VBG ： 动词不定式，过去式，过去分词，现在第三人称单数，现在非第三人称，动名词或现在分词
                list_temp[0] += (str(word) + " ")
            if word.tag_ == "VB" or word.tag_ == "VBD" or word.tag_ == "VBN" or word.tag_ == "VBZ" or word.tag_ == "VBP" or word.tag_ == "VBG":
                list_temp[1] += (str(word) + " ")
        fs.append(list_temp)
        out = open('English1_Noun_feature.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(list_temp)
        out.close()
        print(fs)
        return fs
    
    def get_feature_num(self):
        return 1


class SpacyNounExtractEnglish2(Extractor):
    def __init__(self, config_fp, language = "en"):
        Extractor.__init__(self, config_fp)
        # should be modified
        self.language = language
    # should be modified
    
    def extract_row(self, row):
        # input sentences
        q1 = str(row['english_sentence2'])
        # extract English
        nlp = spacy.load("en")
        
        doc1 = nlp(q1)
        
        feature_Noun = self.extract_Noun(doc1)
        
        # return a list which contain Subject Verb Object
        return feature_Noun
    
    def extract_Noun(self, doc):
        # Extract function in low level
        
        arcs = set()
        fs = list()
        
        all_tags = {w.pos: w.pos_ for w in doc}
        list_temp = ["", ""]
        # all tags of first sentence of our document
        for word in list(doc.sents)[0]:
            if word.tag_ == "NN" or word.tag_ == "NNS" or word.tag_ == "NNP" or word.tag_ == "NNPS":
                #NN，NNS，NNP，NNPS ： 名词，名词复数，专有名词，专有名词复数
                #VB，VBD，VBN，VBZ，VBP，VBG ： 动词不定式，过去式，过去分词，现在第三人称单数，现在非第三人称，动名词或现在分词
                list_temp[0] += (str(word) + " ")
            if word.tag_ == "VB" or word.tag_ == "VBD" or word.tag_ == "VBN" or word.tag_ == "VBZ" or word.tag_ == "VBP" or word.tag_ == "VBG":
                list_temp[1] += (str(word) + " ")
        fs.append(list_temp)
        out = open('English2_Noun_feature.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(list_temp)
        out.close()
        print(fs)
        return fs
    
    def get_feature_num(self):
        return 1


class SpacyNounExtractSpanish(Extractor):
    def __init__(self, config_fp, language = "en"):
        Extractor.__init__(self, config_fp)
        # should be modified
        self.language = language
    # should be modified
    
    def extract_row(self, row):
        # input sentences
        qe1 = str(row['spanish_sentense2'])
        # extract English
        nlp = spacy.load("en")
        doc1 = nlp(qe1)
        # doc2 = nlp(qe2)
        
        feature_Noun = self.extract_Noun(doc1)
        
        # return a list which contain Subject Verb Object
        return feature_Noun
    
    def extract_Noun(self, doc):
        # Extract function in low level
        
        arcs = set()
        fs = list()
        
        all_tags = {w.pos: w.pos_ for w in doc}
        
        # all tags of first sentence of our document
        for word in list(doc.sents)[0]:
            if word.tag_ == "NN" or word.tag_ == "NNS" or word.tag_ == "NNP" or word.tag_ == "NNPS" or word.tag_ == "VB" or word.tag_ == "VBD" or word.tag_ == "VBN" or word.tag_ == "VBZ" or word.tag_ == "VBP" or word.tag_ == "VBG":
                #NN，NNS，NNP，NNPS ： 名词，名词复数，专有名词，专有名词复数
                #VB，VBD，VBN，VBZ，VBP，VBG ： 动词不定式，过去式，过去分词，现在第三人称单数，现在非第三人称，动名词或现在分词
                print(word, word.tag_)
                temp = list()
                temp.append(word)
                temp.append(word.tag_)
                fs.append(temp)
            
            return fs

def get_feature_num(self):
    return 1





def demo():
    config_fp = '../conf/featwheel.conf'
    out = open('English1_SVO_feature.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["subject", "verb", "object"])
    out.close()
    out = open('English2_SVO_feature.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["subject", "verb", "object"])
    out.close()
    out = open('English1_Noun_feature.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["noun", "verb"])
    out.close()
    out = open('English2_Noun_feature.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["noun", "verb"])
    out.close()
    SpacySVOExtractEnglish1(config_fp).extract('cikm_english_and_spanish_train_20180516.csv')
    SpacySVOExtractEnglish2(config_fp).extract('cikm_english_and_spanish_train_20180516.csv')
    SpacyNounExtractEnglish1(config_fp).extract('cikm_english_and_spanish_train_20180516.csv')
    SpacyNounExtractEnglish2(config_fp).extract('cikm_english_and_spanish_train_20180516.csv')
    # SpacyNounExtractSpanish(config_fp).extract('cikm_english_and_spanish_train_20180516.csv')
    return

if __name__ == '__main__':
    demo()


nlp = spacy.load("en")
# # nlp = spacy.load("es")
#
IndexedWord = namedtuple('IndexedWord','word index')
DependencyArc = namedtuple('DependencyArc','start_word end_word label')
doc = nlp(u"I eat a fantastic huge delicious orange which I have never seen")
#
arcs = set()
fs = list()
# #
#
# all_tags = {w.pos: w.pos_ for w in doc}
#
# # all tags of first sentence of our document
# for word in list(doc.sents)[0]:
#     print(word, word.tag_)

#
# for token in doc:
#     newArc = DependencyArc(IndexedWord(token.head.text,token.head.i+1),IndexedWord(token.text,token.i+1),token.dep_)
#     arcs.add(token)
#     print(token)
#     print(token.dep_)
#     # if token.dep_ == "nsubj" or token.dep_ == "ROOT" or token.dep_ == "attr" or token.dep_ == "dobj":
#     #     temp = list()
#     #     temp.append(token)
#     #     temp.append(token.dep_)
#     #     fs.append(temp)
# print(fs)
#
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

