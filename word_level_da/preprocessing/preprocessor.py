"""
This class is intended to preprocess raw data
"""

import re
from nltk.tokenize import RegexpTokenizer


class Preprocessor(object):
    FLAGS = re.MULTILINE | re.DOTALL
    def __init__(self):
        self.nltk_tokenizer = RegexpTokenizer(
            "\\#+[\\w_]+[\\w\\'_\\-]*[\\w_]+|@[\\w_]+|[a-zA-Z'ÁÉÍÓÚáéíóúñÑüÜ]+-*[a-zA-Z'ÁÉÍÓÚáéíóúñÑüÜ]+|["
            "a-zA-Z'ÁÉÍÓÚáéíóúñÑüÜ]+|[<>]?[:;=8][\\-o\\*\\']?[\\)\\]\\(\\[oOdDpP/\\:\\}\\{@\\|\\\\3\\*]|[\\)\\]\\(\\["
            "oOdDpP/\\:\\}\\{@\\|\\\\3\\*][\\-o\\*\\']?[:;=8][<>]?|[.]+|[/,$?:;!()&%#=+{}*~.]+")

    def tokenize_reddit(self, text):
        """
        Tokenize and removes punctuation from the raw text
        :param text: raw text
        :return: string object text without punctuation
        """

        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=self.FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " http_ ")
        text1 = self.nltk_tokenizer.tokenize(text.lower())
        text2 = self.clean_text(text1)

        return ' '.join(text2.split())

    @staticmethod
    def clean_text(text):
        """
        Removes punctuation from text
        :return: string object
        """
        cleaned_text = ''
        for element in text:
            cleaned_text += ''.join(ch for ch in element if ch.isalnum() or ch == "_")
            cleaned_text += ' '

        return cleaned_text
