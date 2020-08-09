"""
This class loads the desired dataset.
"""

import os
import configparser
import numpy as np
import word_level_da
from .process_data import ProcessData


class Dataset(object):

    def __init__(self, key=None, doc_len=500, min_len=100, encode=True, chunking=False, remove_end=True):
        self.config = configparser.ConfigParser()
        self.config.sections()

        self.dataset = None
        current_dir = os.path.abspath(word_level_da.__file__)
        complete_dir = current_dir.split("__")[0]
        self.config.read(os.path.join(complete_dir, '.editorconfig'))

        self.dataset = self.config[key]
        self.key = key
        self.doc_len = doc_len
        self.min_len = min_len
        self.encode = encode
        self.chunking = chunking
        self.remove_end = remove_end

    def get_dataset(self, folder_name="", truth_name="train_golden_truth.txt", partition=None, encode=True):
        process_data = ProcessData()

        docs, labels, ids, truths, lens = process_data.load_txt_train(
            os.path.join(self.dataset[partition], folder_name),
            os.path.join(self.dataset[partition], truth_name),
            remove_end=self.remove_end,
            chunking=self.chunking,
            max_len=self.doc_len,
            min_len=self.min_len,
            encode=encode
        )

        if self.chunking:
            return np.array(docs, dtype=object), np.array(labels), ids, (np.array(truths), np.array(lens))
        else:
            return np.array(docs, dtype=object), np.array(truths), ids, ([], [])

    def get_train_test(self, return_ids=False):
        docs_train, lab_training, ids_training, user_data = self.get_dataset(folder_name=self.dataset["training_folder"],
                                                                             truth_name=self.dataset["training_file"],
                                                                             partition="training")
        docs_test, lab_test, ids_test, user_data = self.get_dataset(folder_name=self.dataset["test_folder"],
                                                                    truth_name=self.dataset["test_file"],
                                                                    partition="test")

        if return_ids:
            return (docs_train, lab_training), (docs_test, lab_test, ids_test, user_data)

    def chunking_text(self, string, lenght):
        return (string[0 + i:lenght + i] for i in range(0, len(string), lenght))