# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:53:33 2019

@author: v1kto
"""
import os

from sklearn import svm
from word_level_da.classifier.feature_extraction import FeatureExtraction
from word_level_da.classifier.scores import Score


class Svm_Text(object):
    """
    Args:
    """

    def __init__(self, docs_train, y_train, weighted=False):
        if weighted:
            self.clf = svm.LinearSVC(C=1.0, class_weight='balanced')
        else:
            self.clf = svm.LinearSVC(C=1.0)

        self.docs_train = docs_train
        self.y_train = y_train
        self.X_train = None
        self.X_test = None
        self.predicted = None

    def extract_features(self, docs_test, feature="word", method="tf-idf", nrange1=(1, 1), nrange2=(1, 1), k=None,
                         stop_words=None, norm=None, idf=True, feature_selection=False, reduce_method="IG"):

        self.ft = FeatureExtraction(self.docs_train, method=method, feature=feature, w_range=nrange1,
                                    c_range=nrange2, stop_wors=stop_words, norm=norm, use_idf=idf, k=k)
        self.ft.fit_test(docs_test)

        if feature_selection:
            self.X_train, self.X_test = self.ft.dim_reduction(self.y_train, k, reduce_method)
        else:
            self.X_train = self.ft.X_vec
            self.X_test = self.ft.X_test

        print("# Features: ", self.ft.X_vec.shape)
        print("Training: ", self.X_train.shape)
        print("Testing: ", self.X_test.shape)

    def train_and_test(self, y_test, reference_data=None, umbral=0.5, method="avg"):
        self.clf.fit(self.X_train, self.y_train)
        predicted_chunks = self.clf.predict(self.X_test)

        (truths, lens) = reference_data

        sc = Score(y_truth=y_test, y_pred=predicted_chunks)

        if len(truths) > 0:
            if method == "avg":
                sc.join_pred(truths, lens, umbral)
            if method == "max":
                sc.max_pred(truths, lens, umbral)

        acc = sc.accuracy()
        f1 = sc.f1()

        scores = [self.X_train.shape[1], acc, f1[0], f1[1], f1[2]]
        return scores