# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:09:47 2019

@author: v1kto
"""
from tensorflow.keras.preprocessing import text
import warnings
from datetime import datetime
import fasttext.util
from word_level_da import utils
from word_level_da.preprocessing.load_data import Dataset
from word_level_da.preprocessing.load_data import ProcessData

warnings.filterwarnings("ignore")

FAST300 = "D:/Models/fasttex/cc.en.300.bin"
VOCAB_DIR = r"D:\v1ktop\Drive-INAOE\Code\data_aumentation_for_author_profiling\word_level_da\obj"

if __name__ == "__main__":

    key = "depresion19_local"
    glove_file = FAST300
    batch_size = 64
    len_doc = 64
    max_features = 15 * 10000

    logger = utils.configure_root_logger(prefix_name=key)
    utils.set_working_directory()

    logger.info("Testing %s dataset", key)

    data = Dataset(key=key, doc_len=len_doc, min_len=int(len_doc / 2), chunking=True, remove_end=True)

    methods = ["Base"]

    n_docs = [i for i in range(1, 11)]

    we_name = "WE_" + key.split(" ")[0]
    ft = fasttext.load_model(FAST300)
    folder = "prep_chunks_filtered"
    truth_file = "golden_truth_filtered.txt"

    for augmentation_method in methods:
        for n in n_docs:
            prefix = augmentation_method + str(n)
            # folder = augmentation_method + "/" + prefix
            # truth_file = augmentation_method + "/" + prefix + ".txt"

            docs, l_docs, ids, useless_data = data.get_dataset(folder_name=folder, truth_name=truth_file,
                                                               partition="training")

            # Create vocabulary with training texts.
            tokenizer = text.Tokenizer(num_words=150000)
            tokenizer.fit_on_texts(docs)

            vocab = ProcessData.load_obj(VOCAB_DIR, we_name)

            n_words_in = 0
            for word, i in tokenizer.word_index.items():
                if i >= len(tokenizer.word_index):
                    continue
                if word not in vocab:
                    embedding_vector = ft.get_word_vector(word)
                    vocab[word] = embedding_vector
                    n_words_in += 1

            print("Number of new embeddings: ", n_words_in)

            ProcessData.save_obj(VOCAB_DIR, we_name, vocab)

    logger.info("Finish Time: %s", datetime.now())
