"""
This class handles raw files, xml, txt etc.
"""

import csv
import fnmatch
import logging
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from .preprocessor import Preprocessor


class ProcessData(object):

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
