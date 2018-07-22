# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 00:24
# @Author  : jinruoyu
# @Email   :


from extractor import Extractor

from statistics import TFIDF, TFCount
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# nmf model
class NMFDecomposition(Extractor):
    def __init__(self, config_fp,  n_components=50, tf_idf=None, tf_idf_result=None):
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tf_idf = tf_idf
        self.tf_idf_result = tf_idf_result
        self.nmf_model = self.init_nmf()

    def init_nmf(self):
        nmf = NMF(n_components=self.n_components, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(self.tf_idf_result)
        print('----nmf-------')
        nmf.fit(self.tf_idf_result)
        # svd_result = svd.transform(self.tf_idf_result)
        return nmf

    def extract_row(self, row):
        # get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        # get tf idf of q1 & q2
        tf1 = self.tf_idf.transform([str(q1)])
        tf2 = self.tf_idf.transform([str(q2)])

        # process nmf on q1 & q2
        nmf1 = self.nmf_model.transform(tf1)
        nmf2 = self.nmf_model.transform(tf2)

        # cal similarity of q1 & q2
        # cosine_similarity
        return cosine_similarity(nmf1, nmf2, dense_output=True)[0].tolist()
    def get_feature_num(self):
        return 1


# lda model
class LDADecomposition(Extractor):
    def __init__(self, config_fp, n_components=50, tf=None, tf_result=None):
        # Use tf (raw term count) features for LDA.
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tf,self.tf_result = tf, tf_result
        self.lda_model = self.init_lda()

    def init_lda(self):
        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5)
                                        # learning_method='online',
                                        # learning_offset=50.,
                                        # random_state=0)
        print('----lda-------')
        lda.fit(self.tf_result)
        # lda_result = svd.transform(self.tf_idf_result)
        return lda

    def extract_row(self, row):
        # get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        # get tf idf of q1 & q2
        tf1 = self.tf.transform([str(q1)])
        tf2 = self.tf.transform([str(q2)])

        # process svd on q1 & q2
        lda1 = self.lda_model.transform(tf1)
        lda2 = self.lda_model.transform(tf2)

        # cal similarity of q1 & q2
        # cosine_similarity
        
        return cosine_similarity(lda1, lda2, dense_output=True)[0].tolist()

    def get_feature_num(self):
        return 1


class SVDDecomposition(Extractor):
    def __init__(self, config_fp, n_components = 50, tf_idf=None, tf_idf_result=None):
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tf_idf, self.tf_idf_result = tf_idf, tf_idf_result
        self.svd_model = self.init_svd()

    def init_svd(self):
        svd = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        print('----svd-------')
        svd.fit(self.tf_idf_result)
        # svd_result = svd.transform(self.tf_idf_result)
        return svd

    def extract_row(self, row):
        # get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        # get tf idf of q1 & q2
        tf1 = self.tf_idf.transform([str(q1)])
        tf2 = self.tf_idf.transform([str(q2)])

        # process svd on q1 & q2
        svd1 = self.svd_model.transform(tf1)
        svd2 = self.svd_model.transform(tf2)

        # cal similarity of q1 & q2
        # cosine_similarity
        return cosine_similarity(svd1, svd2, dense_output=True)[0].tolist()
    def get_feature_num(self):
        return 1


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def demo(n_components=50):
    #Need_change
    config_fp = '../conf/featwheel.conf'
    TFIDF_model = TFIDF(config_fp)
    tf_idf, tf_idf_result = TFIDF_model.tfidf, TFIDF_model.tfidf_result
    print("Extracting tf features for LDA...")
    TFCount_model = TFCount(config_fp)
    tf, tf_result = TFCount_model.tf, TFCount_model.tf_result
    SVDDecomposition(config_fp, n_components=n_components, tf_idf=tf_idf, tf_idf_result=tf_idf_result).extract('preprocessing_train_merge.csv')
    NMFDecomposition(config_fp, n_components=n_components, tf_idf=tf_idf, tf_idf_result=tf_idf_result).extract('preprocessing_train_merge.csv')
    LDADecomposition(config_fp, n_components=n_components, tf=tf, tf_result=tf_result).extract('preprocessing_train_merge.csv')
    SVDDecomposition(config_fp, n_components=n_components, tf_idf=tf_idf, tf_idf_result=tf_idf_result).extract('preprocessing_test.csv')
    NMFDecomposition(config_fp, n_components=n_components, tf_idf=tf_idf, tf_idf_result=tf_idf_result).extract('preprocessing_test.csv')
    LDADecomposition(config_fp, n_components=n_components, tf=tf, tf_result=tf_result).extract('preprocessing_test.csv')


if __name__ == '__main__':
    demo(n_components=50)