#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/15 23:04
# @Author  : HouJP
# @Email   : houjp1992@gmail.com
import configparser as ConfigParser
import re
import time
import random
import sys
from difflib import SequenceMatcher
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
    import lzma
    import Levenshtein
except:
    pass


MISSING_VALUE_NUMERIC = -1
class MyUtil(object):
    @staticmethod
    def save_feature(feature, feature_file):
        if isinstance(feature, str):
            feature_file.write('%s\n' % feature)
        elif isinstance(feature, list):
            #change_yj
            feature = ' '.join(['%s:%s' % (kv[0], kv[1]) for kv in enumerate(feature)])
            feature_file.write('%s\n' % feature)
        else:
            feature_file.write('\n')

    def sentenceInSetByPeopelGraphResult(sen):
        graph_result = [[['impuestos']],[['Cómo'],['reporto','enviar','informar','reportar','informo'],['proveedor']],[['hacer','Cómo'],['pedido']],[['bancaria']],[['Quiero'],['pagar']], [['no','ni','nunca'],['pedido']], [['Donde'],['cupones']],[['número'],['teléfono']],[['Recibí'],['pedido']],[['recibí','recibido'],['no','ni','nunca']],[['confiable'],['vendedor','proveedor']],[['protección'],['comprador','compra']],[['mi'],['pregunta']],
        [['no','ni','nunca'],['abrir','abro','Quiero','presento'],['disputa','disputo']],
        [['Cuándo'],['producto']],[['Quiero'],['cancelar','cancelarlo']],[['no','Cómo','Que'],['monedas']],[['eliminar','recuperar','cancelo','elimine','QUITAR'],['tarjeta','cuenta']],
        [['rastrear'],['paquete','pedido']],[['cómo','Quiero','Necesito'],['queja','quejarse','reclamo']],[['ver'],['órdenes','pedido','orden']],[['humano','persona','eres','Es'],['computadora']]]
        words = sen.lower().strip('¿').split()
        for sub_graph in graph_result:
            flag = 0
            for andSection in sub_graph:
                for orWord in andSection:
                    if orWord.lower() in words:
                        flag += 1
                        break
            if flag == len(sub_graph):
                return True
        return False

    def bothSentencesInSameSubSet(sen1,sen2):
        if MyUtil.sentenceInSetByPeopelGraphResult(sen1) and MyUtil.sentenceInSetByPeopelGraphResult(sen2):
            return True
        return  False

    def singleSentencesInSameSubSet(sen1,sen2):
        if (MyUtil.sentenceInSetByPeopelGraphResult(sen1) and not MyUtil.sentenceInSetByPeopelGraphResult(sen2)) or (not MyUtil.sentenceInSetByPeopelGraphResult(sen1) and MyUtil.sentenceInSetByPeopelGraphResult(sen2)):
             return True
        return False

    def noneSentencesInSameSubSet(sen1,sen2):
        if MyUtil.sentenceInSetByPeopelGraphResult(sen1) or MyUtil.sentenceInSetByPeopelGraphResult(sen2):
            return False
        return  True

    @staticmethod
    def bothSentencesInSameSubGraph(graph_result,sen1,sen2):
        for e in graph_result:
            cur_set = graph_result[e]
            if sen1 in cur_set and sen2 in cur_set:
                return True
        return  False

    @staticmethod
    def singleSentencesInSameSubGraph(graph_result,sen1,sen2):
        for e in graph_result:
            cur_set = graph_result[e]
            if (sen1 in cur_set and sen2 not in cur_set) or (sen1 not in cur_set and sen2 in cur_set):
                return True
        return False

    @staticmethod
    def noneSentencesInSameSubGraph(graph_result,sen1,sen2):
        for e in graph_result:
            cur_set = graph_result[e]
            if sen1 in cur_set or sen2 in cur_set:
                return False
        return  True

    @staticmethod
    def getPredictFeature(Flag_trainOrtest = 'train',ShFeature_pt=None,save_pt=None):
        if Flag_trainOrtest == 'train':
            data_feature_fp = save_pt + ShFeature_pt.split('.')[0] + '.preprocessing_train_merge.csv.smat'
            feature_file = open(data_feature_fp, 'w')
            feature_file.write('%d %d\n' % (21400, 1))
        else:
            data_feature_fp = save_pt + ShFeature_pt.split('.')[0] +  '.preprocessing_test.csv.smat'
            feature_file = open(data_feature_fp, 'w')
            feature_file.write('%d %d\n' % (5000, 1))
        sh_feature = open(ShFeature_pt,'r')
        line = sh_feature.readline()
        index = 0
        while line:
            index += 1
            if Flag_trainOrtest == 'train':
                if index <= 21400:
                    feature1 = [line.split(',')[0][0:len(line.split(',')[0])-1]]
                    MyUtil.save_feature(feature1, feature_file)
            if Flag_trainOrtest == 'test':
                if index > 21400:
                    feature1 = [line.split(',')[0][0:len(line.split(',')[0])-1]]
                    MyUtil.save_feature(feature1, feature_file)
            line = sh_feature.readline()
    print("save train and test Predict Feadure done!")


    @staticmethod
    def getFeatureName(config_fp):
        config = ConfigParser.ConfigParser()
        config.read(config_fp)
        result = config.get('FEATURE', 'feature_selected_num_analysis').split()
        features_num = []
        features_name = []
        multi_feature_name = []
        for e in result:
            cur_feature_name = e.split('-')[0]
            cur_feature_num = int(e.split('-')[1])
            features_num.append(cur_feature_num)
            features_name.append(cur_feature_name)
        for e,cur_feature_name in zip(features_num,features_name):
            for i in range(e):
                multi_feature_name.append(cur_feature_name)
        return multi_feature_name

    @staticmethod
    def getWord2VecDict(word2vec_pt):
        word2vec_dict = {}
        with open(word2vec_pt,'r',encoding='utf-8') as f_in:
            for raw_line in f_in:
                line = raw_line.strip('\n\r').split()
                if True:
                    try:
                        word2vec_dict[line[0]] = [float(i) for i in line[1:]]
                    except:
                        continue
        return word2vec_dict


class StrUtil(object):
    """
    Tool of String
    """

    def __init__(self):
        pass

    @staticmethod
    def tokenize_doc_en(doc):
        doc = doc.decode('utf-8')
        token_pattern = re.compile(r'\b\w\w+\b')
        lower_doc = doc.lower()
        tokenize_doc = token_pattern.findall(lower_doc)
        tokenize_doc = tuple(w for w in tokenize_doc)
        return tokenize_doc


class LogUtil(object):
    """
    Tool of Log
    """

    def __init__(self):
        pass

    @staticmethod
    def log(typ, msg):
        """
        Print message of log
        :param typ: type of log
        :param msg: message of log
        :return: none
        """
        print("[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg)))
        sys.stdout.flush()
        return


class TimeUtil(object):
    """
    Tool of Time
    """
    def __init__(self):
        return

    @staticmethod
    def t_now():
        """
        Get the current time, e.g. `2016-12-27 17:14:01`
        :return: string represented current time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def t_now_YmdH():
        return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


class DataUtil(object):
    """
    Tool of data process
    """
    valid_types = ['str', 'int', 'float']

    def __init__(self):
        return
    
    @staticmethod
    def printVectors(vector):
        num_vector = len(vector)
        print(num_vector)
        for i in range(len(vector[0])):
            cur_string = ''
            for j in range(num_vector):
                cur_string += str(vector[j][i])
                cur_string += ' '
            print(cur_string)

    @staticmethod
    def save_dic2csv(dic, header, out_fp):
        """
        Save dict instance to disk with CSV format
        :param dic: dict instance
        :param header: header of CSV file
        :param out_fp: output file path
        :return: none
        """
        fout = open(out_fp, 'w')
        fout.write('%s\n' % header)
        for k in dic:
            fout.write('"%s","%s"\n' % (k, dic[k].replace("\"", "\"\"")))
        fout.close()

    @staticmethod
    def random_split(instances, rates):
        """
        Random split data set with rates
        :param instances: data set
        :param rates: Proportions of each part of the data
        :return: list of subsets
        """
        LogUtil.log("INFO", "random split data(N=%d) into %d parts, with rates(%s) ..." % (
            len(instances), len(rates), str(rates)))
        slices = []
        pre_sum_rates = []
        sum_rates = 0.0
        for rate in rates:
            slices.append([])
            pre_sum_rates.append(sum_rates + rate)
            sum_rates += rate
        for instance in instances:
            randn = random.random()
            for i in range(0, len(pre_sum_rates)):
                if randn < pre_sum_rates[i]:
                    slices[i].append(instance)
                    break
        n_slices = []
        for slic in slices:
            n_slices.append(len(slic))
        LogUtil.log("INFO", "random split data done, with number of instances(%s)." % (str(n_slices)))
        return slices

    @staticmethod
    def load_vector(file_path, ele_type):
        """
        Load vector from disk
        :param file_path: vector file path
        :param ele_type: element type in vector
        :return: a vector in List type
        """
        assert ele_type.lower() in DataUtil.valid_types, "Wrong ele_type: %s" % ele_type
        ele_type = eval(ele_type.lower())
        vector = []
        f = open(file_path)
        for line in f:
            value = ele_type(line.strip())
            vector.append(value)
        f.close()
        LogUtil.log("INFO", "load vector done. length=%d" % (len(vector)))
        return vector

    @staticmethod
    def save_vector(file_path, vector, mode):
        """
        Save vector on disk
        :param file_path: vector file path
        :param vector: a vector in List type
        :param mode: mode of writing file
        :return: none
        """
        file = open(file_path, mode, encoding='utf-8')
        for value in vector:
            file.write(str(value) + "\n")
        file.close()
        return

    @staticmethod
    def load_matrix(file_path):
        """
        Load matrix from disk
        :param file_path: matrix file path
        :return: a matrix in 2-dim List type
        """
        matrix = []
        file = open(file_path)
        for line in file:
            vector = line.strip().split(',')
            vector = [float(vector[i]) for i in range(len(vector))]
            matrix.append(vector)
        file.close()
        LogUtil.log("INFO", "load matrix done. size=(%d,%d)" % (len(matrix), len(matrix[0])))
        return matrix

    @staticmethod
    def save_matrix(file_path, instances, mode):
        """
        Save matrix on disk
        :param file_path: matrix file path
        :param instances: a matrix in 2-dim List type
        :param mode: mode of writing file
        :return: none
        """
        file = open(file_path, mode)
        for instance in instances:
            file.write(','.join([str(instance[i]) for i in range(len(instance))]))
            file.write('\n')
        file.close()
        return


class MathUtil(object):
    """
    Tool of Math
    """

    @staticmethod
    def count_one_bits(x):
        """
        Calculate the number of bits which are 1
        :param x: number which will be calculated
        :return: number of bits in `x`
        """
        n = 0
        while x:
            n += 1 if (x & 0x01) else 0
            x >>= 1
        return n

    @staticmethod
    def int2binarystr(x):
        """
        Convert the number from decimal to binary
        :param x: decimal number
        :return: string represented binary format of `x`
        """
        s = ""
        while x:
            s += "1" if (x & 0x01) else "0"
            x >>= 1
        return s[::-1]

    @staticmethod
    def try_divide(x, y, val=0.0):
        """
        try to divide two numbers
        """
        if y != 0.0:
            val = float(x) / y
        return val

    @staticmethod
    def corr(x, y_train):
        """
        Calculate correlation between specified feature and labels
        :param x: specified feature in numpy
        :param y_train: labels in numpy
        :return: value of correlation
        """
        if MathUtil.dim(x) == 1:
            corr = pearsonr(x.flatten(), y_train)[0]
            if str(corr) == "nan":
                corr = 0.
        else:
            corr = 1.
        return corr

    @staticmethod
    def dim(x):
        d = 1 if len(x.shape) == 1 else x.shape[1]
        return d

    @staticmethod
    def aggregate(data, modes):
        valid_modes = ["size", "mean", "std", "max", "min", "median"]

        if isinstance(modes, str):
            assert modes.lower() in valid_modes, "Wrong aggregation_mode: %s" % modes
            modes = [modes.lower()]
        elif isinstance(modes, list):
            for m in modes:
                assert m.lower() in valid_modes, "Wrong aggregation_mode: %s" % m
                modes = [m.lower() for m in modes]
        aggregators = [getattr(np, m) for m in modes]

        aggeration_value = list()
        for agg in aggregators:
            try:
                s = agg(data)
            except ValueError:
                s = MISSING_VALUE_NUMERIC
            aggeration_value.append(s)
        return aggeration_value

    @staticmethod
    def cut_prob(p):
        p[p > 1.0 - 1e-15] = 1.0 - 1e-15
        p[p < 1e-15] = 1e-15
        return p

    @staticmethod
    def logit(p):
        assert isinstance(p, np.ndarray), 'type error'
        p = MathUtil.cut_prob(p)
        return np.log(p / (1. - p))

    @staticmethod
    def logistic(y):
        assert isinstance(p, np.ndarray), 'type error'
        return np.exp(y) / (1. + np.exp(y))


class DistanceUtil(object):
    """
    Tool of Distance
    """

    @staticmethod
    def edit_dist(str1, str2):
        try:
            # very fast
            # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
            import Levenshtein
            d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
        except:
            # https://docs.python.org/2/library/difflib.html
            d = 1. - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
        return d

    @staticmethod
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

    @staticmethod
    def is_str_match(str1, str2, threshold=1.0):
        assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
        if float(threshold) == 1.0:
            return str1 == str2
        else:
            return (1. - DistanceUtil.edit_dist(str1, str2)) >= threshold

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def compression_dist(x, y, l_x=None, l_y=None):
        if x == y:
            return 0
        x_b = x.encode('utf-8')
        y_b = y.encode('utf-8')
        if l_x is None:
            l_x = len(lzma.compress(x_b))
            l_y = len(lzma.compress(y_b))
        l_xy = len(lzma.compress(x_b + y_b))
        l_yx = len(lzma.compress(y_b + x_b))
        dist = MathUtil.try_divide(min(l_xy, l_yx) - min(l_x, l_y), max(l_x, l_y))
        return dist

    @staticmethod
    def cosine_sim(vec1, vec2):
        try:
            s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        except:
            try:
                s = cosine_similarity(vec1, vec2)[0][0]
            except:
                s = MISSING_VALUE_NUMERIC
        return s

    @staticmethod
    def jaccard_coef(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(float(len(A.intersection(B))), len(A.union(B)))

    @staticmethod
    def dice_dist(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(2. * float(len(A.intersection(B))), (len(A) + len(B)))


class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([str(words[i]), str(words[i + k])]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([str(words[i]), str(words[i + k1]), str(words[i + k1 + k2])]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in xrange(L - 2):
                for j in xrange(i + 1, L - 1):
                    for k in xrange(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
            Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                for j in xrange(i + 1, L - 2):
                    for k in xrange(j + 1, L - 1):
                        for l in xrange(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)