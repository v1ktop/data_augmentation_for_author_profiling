"""Module to explore data.

Contains functions to help study, visualize and understand datasets.
forked from: tensorflow text classification guides

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import multidict as multidict
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


class explore_data(object):

    def __init__(self, samples=list(), labels=list(), plot_dir="", plot_format="png"):
        """

        :param samples: list or array of documents
        :param labels: list of labels
        :param plot_dir: directory to save the generated plots
        :param plot_format: plot format to save


        """
        self.samples = samples
        self.labels = labels
        self.save_plot_dir = plot_dir
        self.format = plot_format

    def get_num_classes(self):
        """Gets the total number of classes.
    
        # Arguments

        # Returns
            int, total number of classes.
    
        # Raises
            ValueError: if any label value in the range(0, num_classes - 1)
                is missing or if number of classes is <= 1.
        """
        num_classes = max(self.labels) + 1
        missing_classes = [i for i in range(num_classes) if i not in self.labels]
        if len(missing_classes):
            raise ValueError('Missing samples with label value(s) '
                             '{missing_classes}. Please make sure you have '
                             'at least one sample for every label value '
                             'in the range(0, {max_class})'.format(
                missing_classes=missing_classes,
                max_class=num_classes - 1))

        if num_classes <= 1:
            raise ValueError('Invalid number of labels: {num_classes}.'
                             'Please make sure there are at least two classes '
                             'of samples'.format(num_classes=num_classes))
        return num_classes

    def get_num_words_per_sample(self):
        """Gets the median number of words per sample given corpus.
    
        # Arguments
            sample_texts: list, sample texts.
    
        # Returns
            int, median number of words per sample.
        """
        num_words = [len(s.split()) for s in self.samples]
        return np.median(num_words)

    def plot_frequency_distribution_of_ngrams(self, ngram_range=(1, 2),
                                              num_ngrams=50):
        """Plots the frequency distribution of n-grams.
    
        # Arguments
            ngram_range: tuple (min, mplt), The range of n-gram values to consider.
                Min and mplt are the lower and upper bound values for the range.
            num_ngrams: int, number of n-grams to plot.
                Top `num_ngrams` frequent n-grams will be plotted.
        """
        # Create args required for vectorizing.
        kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
        }
        vectorizer = CountVectorizer(**kwargs)

        # This creates a vocabulary (dict, where keys are n-grams and values are
        # idxices). This also converts every text to an array the length of
        # vocabulary, where every element idxicates the count of the n-gram
        # corresponding at that idxex in vocabulary.
        vectorized_texts = vectorizer.fit_transform(self.samples)

        # This is the list of all n-grams in the index order from the vocabulary.
        all_ngrams = list(vectorizer.get_feature_names())
        num_ngrams = min(num_ngrams, len(all_ngrams))
        # ngrams = all_ngrams[:num_ngrams]

        # Add up the counts per n-gram ie. column-wise
        all_counts = vectorized_texts.sum(axis=0).tolist()[0]

        # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
        all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
            zip(all_counts, all_ngrams), reverse=True)])
        ngrams = list(all_ngrams)[:num_ngrams]
        counts = list(all_counts)[:num_ngrams]

        idx = np.arange(num_ngrams)

        beingsaved = plt.figure()
        plt.bar(idx, counts, width=0.8, color='b')
        plt.xlabel('N-gramas')
        plt.ylabel('Frecuencias')
        # plt.title('Distribución de n-gramas')
        if num_ngrams < 50:
            plt.xticks(idx, ngrams, rotation=45)
        plt.show()
        beingsaved.savefig(os.path.join(self.save_plot_dir, "freq_dist" + self.format), format=self.format)

    def plot_frequency_distribution_of_word_list(self, ngram_range=(1, 2),
                                                 show_labels=False, by_docs=False, word_list=[]):

        # Create args required for vectorizing.
        kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
            'binary': by_docs
        }
        vectorizer = CountVectorizer(**kwargs)

        # This creates a vocabulary (dict, where keys are n-grams and values are
        # idxices). This also converts every text to an array the length of
        # vocabulary, where every element idxicates the count of the n-gram
        # corresponding at that idxex in vocabulary.
        vectorized_texts = vectorizer.fit_transform(self.samples)

        # This is the list of all n-grams in the index order from the vocabulary.
        all_ngrams = list(vectorizer.get_feature_names())
        # ngrams = all_ngrams[:num_ngrams]

        # Add up the counts per n-gram ie. column-wise
        all_counts = vectorized_texts.sum(axis=0).tolist()[0]

        # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
        all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
            zip(all_counts, all_ngrams), reverse=True)])

        ngrams = []
        counts = []

        for n_gram, count in zip(all_ngrams, all_counts):
            if n_gram in word_list:
                ngrams.append(n_gram)
                counts.append(count)

        idx = np.arange(len(ngrams))

        beingsaved = plt.figure()
        plt.bar(idx, counts, width=0.8, color='b')
        plt.xlabel('N-gramas')
        plt.ylabel('Frecuencias')
        # plt.title('Distribución de n-gramas')
        if show_labels:
            plt.xticks(idx, ngrams, rotation=45)

        plt.show()
        beingsaved.savefig(os.path.join(self.save_plot_dir, "freq_dist_list" + self.format), format=self.format)

    def plot_sample_length_distribution(self):
        beingsaved = plt.figure()
        plt.hist([len(s) for s in self.samples], 50)
        plt.xlabel('Número de palabras en el documento')
        plt.ylabel('Número de documentos')
        # plt.title('Distribución del número de palabras por documento')
        plt.show()
        beingsaved.savefig(os.path.join(self.save_plot_dir, "length_dist" + self.format), format=self.format)

    def plot_class_distribution(self):

        num_classes = self.get_num_classes()
        count_map = Counter(self.labels)
        counts = [count_map[i] for i in range(num_classes)]
        idx = np.arange(num_classes)

        beingsaved = plt.figure()

        x_values = plt.bar(idx, counts, width=0.8, color='b')

        for rect in x_values:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.xlabel('Clase')
        plt.ylabel('Número de documentos')
        # plt.title('Distribución de la clase')

        plt.xticks(idx, idx)
        plt.show()
        beingsaved.savefig(os.path.join(self.save_plot_dir, "class_dist" + self.format), format=self.format)

    def get_frequency_dict_for_text(self, table):
        fullTermsDict = multidict.MultiDict()
        tmpDict = {}

        # making dict for counting frequencies
        for index, row in table.iterrows():
            tmpDict[index] = int(row)

        for key in tmpDict:
            fullTermsDict.add(key, tmpDict[key])
        return fullTermsDict

    def generate_word_cloud(self, top_words, max_words):
        wc = WordCloud(background_color="white", max_words=max_words, mask=None, max_font_size=150,
                       width=1200, height=720)
        wc.generate_from_frequencies(self.get_frequency_dict_for_text(top_words))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        wc.to_file(os.path.join(self.save_plot_dir, "word_cloud" + self.format))
