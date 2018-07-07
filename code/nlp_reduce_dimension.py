# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 00:24
# @Author  : jinruoyu
# @Email   :


from extractor import Extractor

from statistics import TFIDF
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import  cosine_similarity

#nmf
class nmfDecomposition(Extractor):
    def __init__(self, config_fp,  n_components=50):
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tfidf,self.tf_idf_result = TFIDF(config_fp).init_tfidf()
        self.nmf_model = self.init_nmf()

    def init_nmf(self):
        nmf = NMF(n_components=self.n_components, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(self.tf_idf_result)
        print('----nmf-------')
        nmf.fit(self.tf_idf_result)
        # svd_result = svd.transform(self.tf_idf_result)
        return nmf

    def extract_row(self, row):
        #get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        #get tf idf of q1 & q2
        tf1 = self.tfidf.transform([str(q1)])
        tf2 = self.tfidf.transform([str(q2)])

        #process nmf on q1 & q2
        nmf1 = self.nmf_model.transform(tf1)
        nmf2 = self.nmf_model.transform(tf2)

        #cal similarity of q1 & q2
        # cosine_similarity
        fs = list()
        # print(cosine_similarity(nmf1, nmf2, dense_output=True)[0])
        fs.append(cosine_similarity(nmf1, nmf2, dense_output=True)[0])
        return fs
    def get_feature_num(self):
        return 1
#lda
class ldaDecomposition(Extractor):
    def __init__(self, config_fp, n_components=50):
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tfidf,self.tf_idf_result = TFIDF(config_fp).init_tfidf()
        self.lda_model = self.init_lda()

    def init_lda(self):
        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5)
                                        # learning_method='online',
                                        # learning_offset=50.,
                                        # random_state=0)
        print('----lda-------')
        lda.fit(self.tf_idf_result)
        # lda_result = svd.transform(self.tf_idf_result)
        return lda

    def extract_row(self, row):
        #get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        #get tf idf of q1 & q2
        tf1 = self.tfidf.transform([str(q1)])
        tf2 = self.tfidf.transform([str(q2)])

        #process svd on q1 & q2
        lda1 = self.lda_model.transform(tf1)
        lda2 = self.lda_model.transform(tf2)

        #cal similarity of q1 & q2
        # cosine_similarity
        fs = list()
        # print(cosine_similarity(lda1, lda2, dense_output=True)[0])
        fs.append(cosine_similarity(lda1, lda2, dense_output=True)[0])
        return fs
    def get_feature_num(self):
        return 1


class svdDecomposition(Extractor):
    def __init__(self, config_fp, n_components = 50):
        Extractor.__init__(self, config_fp)
        self.n_components = n_components
        self.tfidf, self.tf_idf_result = TFIDF(config_fp).init_tfidf()
        self.svd_model = self.init_svd()

    def init_svd(self):
        svd = TruncatedSVD(n_components=self.n_components, n_iter=7, random_state=42)
        print('----svd-------')
        svd.fit(self.tf_idf_result)
        # svd_result = svd.transform(self.tf_idf_result)
        return svd

    def extract_row(self, row):
        #get q1 & q2
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        #get tf idf of q1 & q2
        tf1 = self.tfidf.transform([str(q1)])
        tf2 = self.tfidf.transform([str(q2)])

        #process svd on q1 & q2
        svd1 = self.svd_model.transform(tf1)
        svd2 = self.svd_model.transform(tf2)

        #cal similarity of q1 & q2
        # cosine_similarity
        fs = list()
        # print(cosine_similarity(svd1, svd2, dense_output=True)[0])
        fs.append(cosine_similarity(svd1, svd2, dense_output=True)[0])
        return fs
    def get_feature_num(self):
        return 1


def demo(n_components=50):
    #Need_change
    config_fp = '../conf/featwheel.conf'

    # svdDecomposition(config_fp, n_components=n_components).extract('cikm_english_train_20180516.csv')
    nmfDecomposition(config_fp, n_components=n_components).extract('cikm_english_train_20180516.csv')
    ldaDecomposition(config_fp, n_components=n_components).extract('cikm_english_train_20180516.csv')


if __name__ == '__main__':
    demo(n_components=50)