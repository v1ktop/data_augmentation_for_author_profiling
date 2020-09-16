# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:27:17 2020

@author: v1kto
"""

from word_level_da.classifier.feature_extraction import FeatureExtraction
from word_level_da.preprocessing.process_data import  ProcessData

class select_docs(object):

    def __init__(self, training, labels, ids, top_words_path, top_words_file):
        self.docs = training
        self.labels = labels
        self.ids = ids
        self.top_words_dir = top_words_path
        self.top_words_name = top_words_file
        self.top_words = ProcessData.load_obj(top_words_path, top_words_file)

    def get_top_words(self, confidence=0.001, k=None, stop=None, idf=True, docs_training=[],
                      labels_training=[], save_words=False):
        ft = FeatureExtraction(docs_training, stop_wors=stop, use_idf=idf)
        top_words_selected = ft.get_chi_2(labels_training, k, p=confidence)
        new_words = dict.fromkeys(top_words_selected, True)
        old_words = set(self.top_words.keys())
        n_new = len(set(new_words.keys())-old_words)

        if save_words:
            ProcessData.save_obj(self.top_words_dir, self.top_words_name, new_words)
            self.top_words = new_words

        return n_new, new_words

    def select_by_ocurrence(self, max_ocurrence=2, obj_label=1):
        cand_ids = []
        cand_labels = []
        cand_docs = []

        for id_doc, label, doc in zip(self.ids, self.labels, self.docs):
            if label == obj_label:
                count = 0
                for w in set(doc.split(" ")):
                    if w in self.top_words:
                        count += 1
                    if count >= max_ocurrence:
                        cand_ids.append(id_doc)
                        cand_labels.append(label)
                        cand_docs.append(doc)
                        break
        return cand_ids, cand_labels, cand_docs

    def select_by_class(self, obj_label=1):
        cand_ids = []
        cand_labels = []
        cand_docs = []

        for id_doc, label, doc in zip(self.ids, self.labels, self.docs):
            if label == obj_label:
                cand_ids.append(id_doc)
                cand_labels.append(label)
                cand_docs.append(doc)

        return cand_ids, cand_labels, cand_docs

    def select_by_ocurrence_test_set(self, max_ocurrence=2):
        cand_ids = []
        cand_labels = []
        cand_docs = []

        for id_doc, label, doc in zip(self.ids, self.labels, self.docs):
            count = 0
            for w in set(doc.split(" ")):
                if w in self.top_words:
                    count += 1
                if count >= max_ocurrence:
                    cand_ids.append(id_doc)
                    cand_labels.append(label)
                    cand_docs.append(doc)
                    break
        return cand_ids, cand_labels, cand_docs


