# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
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
from word_level_da.preprocessing.process_data import ProcessData
from word_level_da.classifier.feature_extraction import FeatureExtraction

# random.seed(1)

MAX_SYM = 20
SYM_THRESHOLD = 0


class SynRep(object):
    # model=None
    vocab = {}
    stop_words = {}
    # lang='en'
    word_list = []
    words_with_vec = 0
    total_words = 0
    POS_OF_WORD = dict()
    WORD_TOPIC_TRANSLATION = dict()
    pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',  # Adjetivos, Adjetivos comparativos y adjetivos superlativos
                'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',  # Sustantivos , plurales, singulares, propios
                'RB': 'r', 'RBR': 'r', 'RBS': 'r',  # Adverbio, compartivo, superlativo
                'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v',
                'VBZ': 'v'}  # Verbos base, pasado , gerundio , tercera persona

    def __init__(self, glove_file=r"D:\Models\glove\glove.6B\glove.6B.100d.txt", lang="en",
                 replace="wordnet", load_embedding=False, load_obj=False, obj_dir=None, voc_name=None,
                 pos_file=None, analogy_file=None,
                 java_path=r"C:\Program Files (x86)\Common Files\Oracle\Java\javapath",
                 jar="D:/Models/stanford-postagger2018/stanford-postagger-3.9.2.jar",
                 model="D:/Models/stanford-postagger2018/models/wsj-0-18-left3words-nodistsim.tagger"):
        """

        :type replace: object
        """
        os.environ["JAVAHOME"] = java_path
        self.reps_by_doc = None
        self.words_by_doc = None
        if load_obj and replace == "glove":
            self.vocab = ProcessData.load_obj(obj_dir, voc_name)

        if load_embedding:
            word2vec_glove_file = get_tmpfile("glove2.txt")
            glove2word2vec(glove_file, word2vec_glove_file)
            self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

        if lang == 'en':
            self.stop_words = dict.fromkeys(stopwords.words('english'), True)

        if lang == 'es':
            self.stop_words = dict.fromkeys(stopwords.words('spanish'), True)

        self.stop_words["user"] = True
        self.stop_words["url"] = True
        self.stop_words["http"] = True
        self.stop_words["tag"] = True
        self.stop_words["end_"] = True
        self.pos_tagger = StanfordPOSTagger(model, jar, encoding="utf-8")

        if load_obj and replace == "relation":
            self.POS_OF_WORD = ProcessData.load_obj(obj_dir, pos_file)
            self.WORD_TOPIC_TRANSLATION = ProcessData.load_obj(obj_dir, analogy_file)

        self.obj_dir = obj_dir
        self.pos_file = pos_file
        self.word_file = analogy_file
        self.vocab_file = voc_name
        self.replace = replace
        self.top_words = {}
        self.rng = np.random.default_rng()

    def augment_post(self, post, num_aug=1, method="Over",
                     doc_index=0, p_select=0.5, p_replace=0.5,
                     from_class=None, to_class=None, mean=10, std=10):

        original_tokens = word_tokenize(post)

        num_words = len(original_tokens)
        self.words_by_doc[doc_index] += num_words
        a_words = []
        if method == "Rel_0" or method == "Rel_1":
            a_words, n_rep = self.without_replacement(original_tokens, num_aug, from_class, to_class,
                                                      mean, std)
        if method == "Thesaurus":
            a_words, n_rep = self.random(original_tokens, num_aug, p_select, p_replace)
        if method == "Xi" or method == "Context_1":
            a_words, n_rep = self.without_replacement(original_tokens, num_aug, mean=p_select, std=std)
        if method == "Over":
            for _ in range(num_aug):
                a_words.append(' '.join(original_tokens))
                n_rep = 0

        self.reps_by_doc[doc_index] += (n_rep / num_aug)

        if len(a_words) == 0:
            print(doc_index)

        return a_words, n_rep

    def random(self, words, n_docs=1, p_select=0.5, p_replace=0.5):
        num_words = len(words)

        n_sr = np.random.geometric(p_select, n_docs)

        augmented_sentences = []
        all_replaced = 0

        # words=[w[0] for w in pos_original_tokens]

        for new_s in range(n_docs):
            new_words = words.copy()

            real_replaced = 0
            iterations = 0

            while real_replaced < n_sr[new_s]:
                random_index = random.randint(0, num_words - 1)
                random_word = words[random_index]
                synonyms = []

                synonyms = self.get_synonyms(random_word)

                num_syn = len(synonyms)

                index_emb = num_syn + 1

                ##No synonim found, iterate again
                if num_syn >= 1:
                    # print(synonyms)
                    while index_emb >= num_syn:
                        index_emb = np.random.geometric(p_replace) - 1

                    if self.replace == "glove":
                        new_word = synonyms[index_emb][0]

                    else:
                        new_word = synonyms[index_emb]

                    new_words[random_index] = new_word
                    real_replaced += 1
                    all_replaced += 1

                iterations += 1

                if iterations > num_words:
                    break

            # print(new_words)
            augmented_sentences.append(' '.join(new_words))

        return augmented_sentences, all_replaced

    def without_replacement(self, words=[], n_docs=1, from_class="", to_classes=["", "", "", "", ""], mean=32, std=10):
        """
        augmented_sentences : TYPE
            DESCRIPTION.
        real_replaced : int
        :return:
        :param p_select:
        :param words:
        :param n_docs:
        :param from_class:
        :type to_classes: object


        """
        num_words = len(words)

        # n_sr = self.rng.normal(mean, std, n_docs)

        n_sr = np.random.geometric(mean, n_docs)

        perm_inx = np.random.permutation(num_words)

        current_idx = 0
        augmented_sentences = []
        all_replaced = 0
        replace_dict = dict()
        to_class = None

        # words=[w[0] for w in pos_original_tokens]

        pos_original_tokens = nltk.pos_tag(words)

        for new_s in range(n_docs):
            new_words = words.copy()
            real_replaced = 0
            tolerance = 0
            words_to_replace = n_sr[new_s]

            if words_to_replace > num_words:
                words_to_replace = num_words

            if self.replace == "relation":
                if len(to_classes) > 1:
                    to_class = to_classes[new_s - len(to_classes)]

                if (len(to_classes) == 1):
                    to_class = to_classes[0]

            while real_replaced < words_to_replace and len(words) > 1:

                token = pos_original_tokens[perm_inx[current_idx]]
                current_word_pos = token[1]
                current_word = token[0]

                if current_word_pos in self.pos_dict and current_word not in self.top_words and current_word \
                        not in self.stop_words:

                    if self.replace == "relation":
                        key = from_class + '-' + to_class
                    else:
                        key = None

                    candidates = self.get_synonyms(current_word, key)

                    if current_word not in replace_dict:
                        replace_dict[current_word] = list()

                    if len(candidates) > 1:

                        for cand in candidates:
                            if self.pos_dict[current_word_pos] in self.pos_list_of(cand) and cand not in \
                                    replace_dict[current_word]:
                                replace_dict[current_word].append(cand)
                                real_replaced += 1
                                all_replaced += 1
                                new_words[perm_inx[current_idx]] = cand
                                break

                current_idx += 1

                if current_idx >= num_words:
                    current_idx = 0
                    tolerance += 1

                if tolerance > 1:
                    break
            augmented_sentences.append(' '.join(new_words))

        return augmented_sentences, all_replaced

    def pos_list_of(self, word):
        if word not in self.POS_OF_WORD:
            self.POS_OF_WORD[word] = [ss.pos() for ss in wordnet.synsets(word)]
        return self.POS_OF_WORD[word]

    def word_list_translation(self, word, to_class):

        key = to_class
        if key not in self.WORD_TOPIC_TRANSLATION:
            self.WORD_TOPIC_TRANSLATION[key] = dict()
        if word not in self.WORD_TOPIC_TRANSLATION[key]:
            try:
                # get vector a , them rest vector b
                label_vector = self.model.get_vector(to_class)
                word_vector = self.model.get_vector(word)
                result_vector = word_vector - label_vector
                self.WORD_TOPIC_TRANSLATION[key][word] = [x[0] for x in
                                                          self.model.similar_by_vector(result_vector, 20)]
                self.words_with_vec += 1
            except KeyError:
                self.WORD_TOPIC_TRANSLATION[key][word] = [(word, 0)]

    def get_synonyms(self, word, key=None):
        synonyms = []
        if self.replace == "wordnet":
            synonyms = self.get_synonyms_wordnet(word)

        if self.replace == "glove":
            if word in self.vocab.keys():
                synonyms = [x[0] for x in self.vocab[word]]

        if self.replace == "relation":
            if word in self.WORD_TOPIC_TRANSLATION[key]:
                synonyms = self.WORD_TOPIC_TRANSLATION[key][word]

        return synonyms

    def get_close_vector(self, word):
        try:
            syn = self.vocab[word]
            self.words_with_vec += 1
        except KeyError:
            try:
                syn = self.model.similar_by_word(word=word, topn=MAX_SYM)
                self.vocab[word] = syn
                self.words_with_vec += 1
            except KeyError:
                self.vocab[word] = [(word, 0)]
        return syn

    def get_synonyms_wordnet(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)

        if '' in synonyms:
            synonyms.remove('')

        return list(synonyms)

    def expand_stop_words(self):
        new_words = []
        for w in self.stop_words.keys():
            try:
                symn = self.model.most_similar(w)
                for s in symn[0:3]:
                    new_words.append(s[0])
            except KeyError:
                continue

        for word in new_words:
            self.stop_words[word] = True

    def build_vocab_xi2(self, docs_train, truths_train, k=100, idf=False):

        extractor = FeatureExtraction(docs_train=docs_train, use_idf=idf)

        self.word_list = extractor.get_chi_2(truths_train, k=k)

    def build_all_vocab(self, docs_train, to_class=None):
        extractor = FeatureExtraction(docs_train=docs_train, method="count")
        vocab = extractor.cv.vocabulary_
        word_list = []
        for w in vocab:
            if w not in self.stop_words and w.isalpha():
                word_list.append(w)

        for i, w in enumerate(word_list):
            if self.replace == "glove":
                self.get_close_vector(w)
            if self.replace == "operation":
                for key in to_class:
                    self.word_list_translation(w, key)

            if i % 500 == 0:
                print("Processed", i + 1)
                print("Remaining", len(word_list) - (i + 1))
                self.save_files()

            self.pos_list_of(w)

        self.save_files()
        return True

    def save_files(self):

        ProcessData.save_obj(self.obj_dir, self.pos_file, self.POS_OF_WORD)

        if self.replace == "operation":
            ProcessData.save_obj(self.obj_dir, self.word_file, self.WORD_TOPIC_TRANSLATION)
        else:
            ProcessData.save_obj(self.obj_dir, self.vocab_file, self.vocab)

    def get_stats_words(self):
        return self.compute_statistics(self.words_by_doc)

    def get_stats_words_rep(self):
        return self.compute_statistics(self.reps_by_doc)

    def compute_statistics(self, vector):
        values = {'min': np.amin(vector), 'max': np.amax(vector), 'range': np.ptp(vector), 'mean': np.mean(vector),
                  'median': np.median(vector), 'std': np.std(vector)}
        return values

    def load_top_words(self, file_name):
        self.top_words = ProcessData.load_obj(self.obj_dir, file_name)
