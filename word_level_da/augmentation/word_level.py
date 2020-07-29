"""
This class define methods to perform data augmentation at word level.
"""

import os
import random
import numpy as np
import nltk
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from preprocesing import process_data_files
from classifier.FeactureExtraction import feature_extraction