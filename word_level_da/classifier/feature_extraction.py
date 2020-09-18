"""
This class is used to extract features from a collection of text

"""


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd


class FeatureExtraction(object):

    def __init__(self, docs_train, method='tf-idf', feature="word", w_range=(1, 1), c_range=(1, 1),
                 k=None, stop_wors="english", norm="l2", use_idf=False):
        self.X_test = None
        self.X_vec = None
        if method == 'count':
            self.cv = CountVectorizer(analyzer=feature, ngram_range=w_range, binary=False, stop_words=stop_wors)
            # self.X_vec=self.cv.fit_transform(docs_train).toarray()Tru
            self.X_vec = self.cv.fit_transform(docs_train)
        if method == 'tf-idf':
            self.cv = TfidfVectorizer(analyzer=feature, ngram_range=w_range, max_features=k, norm=norm,
                                      use_idf=use_idf, stop_words=stop_wors, sublinear_tf=False)
            self.X_vec = self.cv.fit_transform(docs_train)
        if method == 'ensemble':
            char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=c_range, max_df=.98, min_df=0.001,
                                              sublinear_tf=True)
            word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=w_range, max_df=.98, min_df=0.001,
                                              sublinear_tf=True)

            # Build a transformer (vectorizer) pipeline using the previous analyzers
            # *FeatureUnion* concatenates results of multiple transformer objects

            self.cv = ngrams_vectorizer = Pipeline([('feats', FeatureUnion([('word_ngram', word_vectorizer),
                                                                            ('char_ngram', char_vectorizer),
                                                                            ])),
                                                    ])
            self.X_vec = ngrams_vectorizer.fit_transform(docs_train)

    def get_info_gain(self, y_train, threshold=0.001):

        es = dict(zip(self.cv.get_feature_names(),
                      mutual_info_classif(self.X_vec, y_train, discrete_features=True)
                      ))

        values = pd.DataFrame.from_dict(data=es, orient='index')
        values.rename(index=str, columns={0: 'score'}, inplace=True)
        values['score'] = values['score'].round(6)
        top_words = values.sort_values('score', ascending=False)
        return top_words[top_words.score > threshold].index

    def get_chi_2(self, y_train, k=None, p=0.001):

        score, pval = chi2(self.X_vec, y_train)
        res = dict(zip(self.cv.get_feature_names(), zip(score, pval)))
        values = pd.DataFrame(res).T

        values.rename(index=str, columns={0: 'score', 1:'pval'}, inplace=True)
        values['score'] = values['score'].round(6)
        values['pval'] = values['pval'].round(8)

        if k is None:
            top_words = values.sort_values('score', ascending=False)
            return top_words[0:k]
        else:
            top_words = values.sort_values('pval', ascending=True)
            return top_words[top_words.pval<p]


    def get_domain_vocab(self, k):
        array_docs = self.X_vec.toarray()
        voc = dict.fromkeys(self.cv.vocabulary_, 0)
        for w in self.cv.vocabulary_:
            feature_index = self.cv.vocabulary_.get(w)
            voc[w] = array_docs[:, feature_index].sum()

        fce_sorted = pd.DataFrame.from_dict(data=voc, orient='index')
        fce_sorted.rename(index=str, columns={0: 'score'}, inplace=True)
        fce_sorted['score'] = fce_sorted['score'].round(6)
        top_words = fce_sorted.sort_values('score', ascending=False)
        return top_words[0:k].index

    def fit_test(self, X_test):
        # self.X_test=self.cv.transform(X_test).toarray()
        self.X_test = self.cv.transform(X_test)

    def dim_reduction(self, y_train, k=10000, method="IG"):
        if method == "Xi":
            selector = SelectKBest(score_func=chi2, k=k)
        if method == "IG":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)

        X_train_chi2 = selector.fit_transform(self.X_vec, y_train)
        X_test_chi2 = selector.transform(self.X_test)
        return X_train_chi2, X_test_chi2
