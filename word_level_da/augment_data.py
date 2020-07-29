"""
Execute this script to augment your data locally

"""

import utils
import os
import nltk
import numpy as np
from collections import defaultdict
from preprocesing.load_datasets import Dataset
from preprocesing import process_data_files
from augmentation.syn_rep import SynRep
from augmentation.select_candidates import select_docs

GLOVE_DIR= "D:/Models/glove/glove.42B/glove.42B.300d.txt"


def augment_by_docs_one_class(lan, output, glove_file, method="Over",
                              replace="glove",
                              label_to_aug=None, labels=None, obj_label=1,
                              n_docs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dataset_key="erisk18_dev",
                              load_emb=True, load_obj=False, preproces_vocab=False,
                              vocab_dir="D:/v1ktop/Drive/REPOS/augmentation_ap/obj/",
                              analogy_file="file",
                              p_confidence=0.001, min_ocurrence=20, doc_len=64, p_aug=0.1, mean=32, std=10
                              ):
    logger.info("Loading positive documents")

    data = Dataset(key=dataset_key, encode=False, remove_end=False, doc_len=doc_len,
                   min_len=int(doc_len / 2), chunking=True)

    ##Get dataset in chunks
    docs_train, truths_train, author_ids_train, (truths, lens) = data.get_dataset(partition="training",
                                                                                  folder_name="prep_chunks_joined",
                                                                                  truth_name="train_golden_truth_joined.txt",
                                                                                  tag=False)

    selection = select_docs(docs_train, truths_train, author_ids_train)
    selection.get_top_words(confidence=p_confidence)
    cand_ids, cand_labels, cand_docs = selection.select_by_ocurrence(max_ocurrence=min_ocurrence, obj_label=obj_label)

    logger.info("Number of selected docs %d", len(cand_ids))

    logger.info("Loading glove")

    syn_augmented = SynRep(glove_file=glove_file, lang=lan, replace=replace,
                           load_embedding=load_emb, load_obj=load_obj, obj_dir=vocab_dir,
                           voc_name=dataset_key, pos_file="pos_" + dataset_key, analogy_file=analogy_file)

    if method == "Xi":
        # logger.info("Computing Xi squared of the whole vocabulary")
        # syn_augmented.build_vocab_xi2(docs_train, truths_train,k=top_features, idf=True)
        syn_augmented.word_list = selection.top_words

    logger.info("Top features: %s", selection.top_words)
    logger.info("Number of Top features: %s", len(selection.top_words))

    logger.info("Augmenting docs")

    if preproces_vocab:
        logger.info("Loading vocabulary")

        syn_augmented.build_all_vocab(cand_docs, label_to_aug)

        logger.info("Loading vocabulary successful")
        logger.info("Vocabulary without stop words %d", syn_augmented.total_words)
        logger.info("Number of words with vectors %d", syn_augmented.words_with_vec)

    # new_training=[]

    uniques_ids = defaultdict(list)

    # One class only

    for c_id, c_label, c_doc in zip(cand_ids, cand_labels, cand_docs):
        uniques_ids[c_id].append(c_doc)

    ids_to_aug = []
    truths_to_aug = [obj_label] * len(uniques_ids)

    for num_aug in n_docs:
        syn_augmented.reps_by_doc = np.zeros(len(cand_docs))
        syn_augmented.words_by_doc = np.zeros(len(cand_docs))
        augmented_docs = []
        n_augs = []
        c = 0
        for i, key in enumerate(uniques_ids.keys()):

            docs = uniques_ids[key]
            ids_to_aug.append(key)
            print(ids_to_aug[i])
            # lines=doc.split('end_')

            new_docs = []
            n_augs.append(num_aug)

            current_label = labels[truths_to_aug[i]]
            # print(n_docs)
            for doc in docs:

                if method == "Rel_1" or method == "Rel_0":
                    new_chunks, nrep = syn_augmented.augment_post(post=doc,
                                                                  num_aug=num_aug, method=method, doc_index=i,
                                                                  from_class=current_label,
                                                                  to_class=label_to_aug[current_label], p_select=p_aug,
                                                                  mean=mean, std=std)

                else:
                    new_chunks, nrep = syn_augmented.augment_post(post=doc,
                                                                  num_aug=num_aug, method=method, doc_index=i,
                                                                  from_class=None,
                                                                  to_class=None, p_select=p_aug, mean=mean, std=std)

                new_docs.append(new_chunks)

            for l in range(num_aug):
                single = []
                for m in range(len(new_docs)):
                    # print(m)
                    # print(l)
                    single.append(new_docs[m][l])

                augmented_docs.append(single)

            c += num_aug
            if c % 10 == 0:
                print('Augmented:' + str(c))

        prefix = method + str(num_aug)

        complete_out_dir = os.path.join(output, method)

        if preproces_vocab:
            syn_augmented.save_files()

        labels_to_save = ["1"] * len(uniques_ids)

        new_ids = process_data_files.write_labels(ids_to_aug, labels_to_save, n_augs, complete_out_dir, prefix,
                                                  dataset=dataset_key)

        process_data_files.plain_docs_to_txt(new_ids, augmented_docs, complete_out_dir, prefix)

        stats_words = syn_augmented.get_stats_words()
        stats_rep = syn_augmented.get_stats_words_rep()
        # logger.info("Number of words by doc: %s", stats_words)
        # logger.info("Number of words replaced: %s", stats_rep)
        data = [dataset_key, method, num_aug, stats_words["mean"], stats_rep["mean"]]
        logger.info(data)


if __name__ == "__main__":
    """
    Over: Oversamplig
    Thesaurus: 
    Context_1
    Context_0
    Xi:
    """
    method = "Xi"
    dataset_key = "erisk18_dev"
    # dataset_key="anorexia18_dev"
    lang = 'en'

    labels_dic = {"depressed": ["anxious", "frustrated", "unhappy", "despondent", "discouraged"]}
    labels = {0: "happiness", 1: "depressed"}
    # labels={0:"healthy", 1:"anorexic"}
    # labels_dic={"healthy":["bulimic", "underweight", "obese", "malnourished", "unhealthy"]}

    output_dir = "D:/corpus/DepresionEriskCollections/2017/train/augmented_both"
    # output_dir="D:/corpus/anorexia/2018/train/augmented_normal/"
    logger = utils.configure_root_logger(prefix_name=method + "_" + dataset_key)
    utils.set_working_directory()

    logger.info("Running data augmentation for the dataset: %s", dataset_key)
    logger.info("Method: %s", method)
    logger.info("Labels: %s", labels_dic[labels[1]])

    p_select = 0.2

    """
        replace:
                wordnet: for thesaurus method
                glove: for Xi
                analogy: for Context_1, Context_0
    """

    augment_by_docs_one_class(lan=lang, output=output_dir,
                       glove_file=GLOVE_DIR_TEST,
                       label_to_aug=labels_dic,
                       labels=labels, method=method, replace="glove",
                       n_docs=[i for i in range(1, 11)],
                       dataset_key=dataset_key, load_emb=False, load_obj=True, preproces_vocab=False,
                       analogy_file="l0_word_" + dataset_key, p_aug=p_select, min_ocurrence={0: 25, 1: 20}, mean=6,
                       std=2
                       )
