# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:14:22 2019

@author: v1kto
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class Score(object):
    def __init__(self, y_truth, y_pred):
        self.y_pred = y_pred
        self.y_truth = y_truth
        self.scores = []
        self.probabilities = []

    """
    Binary classification only
    """

    def join_pred(self, truths, lens, umbral=0.5):

        current = 0
        real_pred = np.zeros(len(truths))
        probs = np.zeros(len(truths))

        for k, step in enumerate(lens):
            sum_pred = 0
            this_pred = np.zeros(step)

            for i in range(step):
                sum_pred += self.y_pred[current]
                this_pred[i] = self.y_pred[current]
                current += 1

            real_p = np.mean(this_pred)
            if real_p > umbral:
                p_lab = 1
            else:
                p_lab = 0
            probs[k] = real_p
            real_pred[k] = p_lab

        self.y_pred = real_pred
        self.probabilities = probs
        # Encoder = LabelEncoder()
        self.y_truth = truths

    def max_pred(self, truths, lens, umbral=0.5, q=0.98):

        current = 0
        real_pred = np.zeros(len(truths))
        probs = np.zeros(len(truths))

        for k, step in enumerate(lens):
            predicted_by_user = np.zeros(step)
            for i in range(step):
                predicted_by_user[i] = self.y_pred[current]
                current += 1

            real_p = np.percentile(predicted_by_user, q)
            if (real_p > umbral):
                p_lab = 1
            else:
                p_lab = 0
            probs[k] = real_p
            real_pred[k] = p_lab

        self.y_pred = real_pred
        self.probabilities = probs
        # Encoder = LabelEncoder()
        self.y_truth = truths

    def f1(self):
        F1 = precision_recall_fscore_support(self.y_truth, self.y_pred, average='binary')
        return F1

    def accuracy(self):
        return accuracy_score(self.y_truth, self.y_pred)
