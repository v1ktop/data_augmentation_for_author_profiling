"""
This class handles raw files, xml, txt etc.
"""

import fnmatch
import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from .preprocessor import Preprocessor


class ProcessData(object):

    def load_txt_train(self, docs_dir, truth_dir, remove_end=True, chunking=False,
                       max_len=500, min_len=1, encode=True):
        """


        Parameters
        ----------
        docs_dir : TYPE
            main files directory.
        truth_dir : TYPE
            main truth file directory.
        remove_end : TYPE, optional
            If True remove the label end_ from the text. The default is True.
        chunking : TYPE, optional
            DESCRIPTION. The default is False.
        max_len : TYPE, optional
            DESCRIPTION. The default is 500.
        min_len : TYPE, optional
            Minimun length of a document. The default is 100.
        encode : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        docs_authors : list
            the documents of the dataset
        new_truths: list
            the new truths values if chunking
        author_ids : list
            the id of every document.
        truths: list
            the original real truth after chunking.
        doc_lens :  list
            the number of chunks by document

        """

        doc_lens = []
        new_truths = []
        new_ids = []
        docs_authors = []

        author_ids, truths = self.read_golden_truth(truth_dir, encode=True)
        for i, author_id in enumerate(author_ids):
            filename = author_id + ".txt"
            file = open(os.path.join(docs_dir, filename))
            text = file.read()

            if remove_end:
                text = text.replace("end_", "")

            if chunking:
                posts = self.chunking_text(text.split(" "), max_len)
                n_post = 0
                for post in posts:
                    if len(post) > min_len:
                        n_post += 1
                        docs_authors.append(" ".join(post))

                        new_truths.append(truths[i])
                        new_ids.append(author_ids[i])

                doc_lens.append(n_post)
            else:
                docs_authors.append(text)

            file.close()

        if chunking:
            return docs_authors, new_truths, new_ids, truths, doc_lens
        else:
            return docs_authors, new_truths, author_ids, truths, doc_lens

    @staticmethod
    def read_golden_truth(truth_path, separator="\t", encode=False):
        """Load the truth

        This function reads the truth from the TXT file

        Args:
            truth_path: The path of the truth file.
            separator: the character used to separate the id from the label

        Returns:
            The a sorted list of ids and labels

        """

        temp_sorted_ids = []
        temp_truths = []

        with open(truth_path, 'r') as truth_file:
            for line in sorted(truth_file):
                line = line.rstrip('\n')
                row = line.split(separator)
                temp_sorted_ids.append(row[0])
                if encode:
                    temp_truths.append(int(row[1]))
                else:
                    temp_truths.append(row[1])

        return temp_sorted_ids, temp_truths

    @staticmethod
    def chunking_text(string, lenght):
        return (string[0 + i:lenght + i] for i in range(0, len(string), lenght))

    @staticmethod
    def load_xml_files_erisk(local_dir, token_position=0):
        """
        This method loads xmls files to plain text.
        :param local_dir: Main directory to search for files
        :param token_position: The index that indicates the user name
        for example train_user58 token_position=1
        :return: a dictionary of users[username]: document
        """
        users = {}
        prep = Preprocessor()
        c = 0
        for dir_path, dir_names, filenames in os.walk(local_dir):
            for name in filenames:
                tok = name.split("_")
                if token_position > 0:
                    key = tok[0] + tok[token_position]
                else:
                    key = tok[token_position]
                    key = key.strip(".xml")
                full_file = os.path.abspath(os.path.join(dir_path, name))
                dom = ET.parse(full_file, parser=ET.XMLParser(encoding="utf-8"))
                writing = dom.findall('WRITING')
                for w in writing:
                    title = w.find('TITLE').text
                    text = w.find('TEXT').text
                    post = title + " " + text
                    # preprocess text
                    new_text = prep.tokenize_reddit(post)

                    if key in users.keys():
                        users[key] += new_text + ' end_ '
                    else:
                        users[key] = new_text + ' end_ '

            c += 1
            print("Preprocessed chunk: ", c)

        return users

    @staticmethod
    def users_dict_to_txt(destination_directory, users):
        # Create the directory if it does not exist.
        os.makedirs(destination_directory, exist_ok=True)
        for user in users.keys():
            # Create a txt file with the user ID as the filename (same as the XML files)
            with open(os.path.join(destination_directory, user + ".txt"),
                      'w', encoding="utf-8") as txt_output_file:
                txt_output_file.write(users[user])
        return True

    @staticmethod
    def write_labels(docs_ids, labels, save_dir, file_name="truth_labels.txt"):
        os.makedirs(save_dir, exist_ok=True)
        string_labels = []
        for docs_id, label in zip(docs_ids, labels):
            chain = docs_id + "\t" + str(label)
            string_labels.append(chain)

        with open(os.path.join(save_dir, file_name),
                  'w', encoding="utf-8") as txt_output_file:
            txt_output_file.write('\n'.join(string_labels))
        return True

    @staticmethod
    def plain_docs_to_txt(author_ids, list_docs, destination_directory, prefix):
        txts_destination_directory = os.path.join(destination_directory, prefix)

        # Create the directory if it does not exist.
        os.makedirs(txts_destination_directory, exist_ok=True)

        # Iterate over authors in the test set, docs is list of list
        for author_id, docs in zip(author_ids, list_docs):
            doc = " ".join(docs)
            with open(os.path.join(txts_destination_directory, author_id + ".txt"),
                      'w', encoding="utf-8") as txt_output_file:
                txt_output_file.write(doc)

    @staticmethod
    def save_obj(dir_path, name, obj):
        os.makedirs(dir_path, exist_ok=True)
        obj_path = os.path.join(dir_path, name)
        with open(obj_path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(dir_path, name):
        os.makedirs(dir_path, exist_ok=True)
        try:
            obj_path = os.path.join(dir_path, name)
            with open(obj_path + ".pkl", 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return dict()