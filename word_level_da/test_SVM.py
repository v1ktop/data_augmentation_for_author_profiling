"""
This script test the model with: training - test, partitions
"""
import time
from word_level_da import utils
from word_level_da.preprocessing.load_data import Dataset
from word_level_da.classifier.svm_text import Svm_Text
import numpy as np

if __name__ == "__main__":

    # Parameters
    # key="erisk18_test"
    key = "depresion18_local"
    count = "tf-idf"
    feature = 'word'
    len_doc = 64

    methods = ["Xi"]
    stop_w = None
    idf = True
    score_method = "avg"

    logger = utils.configure_root_logger(prefix_name=key + "_" + methods[0])
    utils.set_working_directory()

    logger.info("Testing %s dataset", key)
    logger.info("Method, %s", count)
    logger.info("Weighted, %s", "All")
    logger.info("Stop words, %s", stop_w)
    logger.info("IDF, %s", idf)
    logger.info("Augmentation method  %s", methods[0])

    data = Dataset(key=key, doc_len=len_doc, min_len=int(len_doc / 2), chunking=True, remove_end=True)
    training, test = data.get_train_test(return_ids=True)

    umbral = 0.5
    n_docs = [i for i in range(1, 11)]

    for augmentation_method in methods:
        for n in n_docs:

            prefix = augmentation_method + str(n)
            folder = augmentation_method + "/" + prefix
            truth_file = augmentation_method + "/" + prefix + ".txt"

            training_a = data.get_dataset(folder_name=folder, truth_name=truth_file, partition="augmented")

            training_augmented = np.append(training[0], training_a[0])
            truths_augmented = np.append(training[1], training_a[1])

            w_range = (1, 1)
            c_range = (0, 0)

            for weighted in [0, 1]:

                svm = Svm_Text(training_augmented, truths_augmented, weighted)
                svm.extract_features(test[0], feature=feature, method=count, nrange1=w_range, nrange2=c_range,
                                     k=None, stop_words=stop_w, norm="l2", idf=idf, feature_selection=False)

                score = svm.train_and_test(test[1], reference_data=(test[3]),
                                           method=score_method, umbral=umbral)

                clas_n = "SVM"

                if weighted:
                    clas_n = "SVM-C"

                B = [clas_n, augmentation_method, n]
                B += score
                logger.info(B)
