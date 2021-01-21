"""
All functions for reviews preprocessing and vectorization.
"""
from nltk.corpus import PlaintextCorpusReader
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams
from nltk import pos_tag
from collections import OrderedDict

from sklearn.utils import shuffle
from multiprocessing import Pool
import numpy as np
from scipy.sparse import csr_matrix
import os
from functools import partial
from tqdm import tqdm


def prepare_train_data(train_files, vec_type='freq'):
    """
    Vectorize train data:
    - clean from service words
    - choose 10000 words for each label that are the most frequent
    - vectorize by them
    - return both vectorized data and important words 
       as they are needed on test phase
    - if type is tf-idf, also return inverse document frequency
    """
    print('-- Train data preparation --')
    print('Cleaning from service words...')
    # leave only nouns, verbs, adjectives and adverbs as only they have sence
    data = clean_reviews(train_files)
    # get 10000 most frequent words for each label
    important_words = get_important_words(data)
    print('Important words were chosen.')
    print('Vectorization...')
    if vec_type == 'freq':
        return vectorize_freq(data, important_words), important_words, None
    else:
        X, Y, idf = vectorize_tf_idf(data, important_words)
        return (X, Y), important_words, idf


def prepare_test_data(test_files, important_words, idf=None, vec_type='freq'):
    """
    Vectorize test data using vocabular got during training.
    For TF-IDF inverse document frequency also is used.
    """
    print('-- Test data preparation --')
    print('Cleaning from service words...')
    # clean reviews from service words
    data = clean_reviews(test_files)
    print('Vectorization...')
    if vec_type == 'freq':
        return vectorize_freq(data, important_words)
    else:
        return vectorize_tf_idf(data, important_words, idf)


def vectorize_freq(data, important_words):
    """
    Vectorize data by frequency.
    """
    # create list of pairs of type:
    # [(list_of_review_words, class_label)]
    labels = data.keys()
    labeled_data = []
    for label in labels:
        for document in data[label]['word_matrix']:
            labeled_data.append((document, label))

    # prepare dense matrix
    matrix_vec = csr_matrix((len(labeled_data), len(
        important_words)), dtype=np.int8).toarray()
    # list of target, ground-truth labels
    target = np.array([' '*10 for _ in np.arange(len(labeled_data))])
    for index_doc, document in enumerate(tqdm(labeled_data)):
        for index_word, word in enumerate(important_words):
            # count frequency inside one review
            matrix_vec[index_doc, index_word] = document[0].count(word)
        target[index_doc] = document[1]
    # shuffle dataset
    X, Y = shuffle(matrix_vec, target)
    return X, Y


def vectorize_tf_idf(data, important_words, idf=None):
    """
    Victorize data with tf-idf.
    Passed IDF is used (if None, it is training phase and IDF will be calculated).
    """
    # create list of pairs of type:
    # [(list_of_review_words, class_label)]
    labels = data.keys()
    labeled_data = []
    for label in labels:
        for document in data[label]['word_matrix']:
            labeled_data.append((document, label))

    # prepare dense matrix
    matrix_vec = csr_matrix((len(labeled_data), len(
        important_words)), dtype=np.int8).toarray()
    # list of target, ground-truth labels
    target = np.array([' '*10 for _ in np.arange(len(labeled_data))])
    for index_doc, document in enumerate(tqdm(labeled_data)):
        for index_word, word in enumerate(important_words):
            # count frequency inside one review
            matrix_vec[index_doc, index_word] = document[0].count(word)
        target[index_doc] = document[1]
    # normalize term frequency
    matrix_vec_sum = matrix_vec.sum(axis=1)[:, np.newaxis]
    matrix_vec_sum[matrix_vec_sum == 0] = 1
    tf = matrix_vec / matrix_vec_sum

    # estimate inverse edocument frequency (if needed)
    return_idf = False
    if idf is None:
        return_idf = True
        a = (matrix_vec > 0).sum(axis=0)
        idf = len(labeled_data) / (matrix_vec > 0).sum(axis=0)
        idf.data = np.log(idf.data)
    tf_idf = tf * idf
    # shuffle dataset
    X, Y = shuffle(tf_idf, target)
    if return_idf:
        return X, Y, idf
    else:
        return X, Y


def get_important_words(data):
    """
    Choose 10000 most frequent words and bigrams 
    from all words in all reviews (by labels).
    """
    all_words = []
    labels = data.keys()
    for label in labels:
        frequency = FreqDist(data[label]['all_words'])
        common_words = frequency.most_common(10000)
        words = [i[0] for i in common_words]
        all_words.extend(words)
    # remove repeated words
    unique_words = list(OrderedDict.fromkeys(all_words))
    return unique_words


def clean_reviews(files):
    """
    Clean reviews from service words.
    Done in parallel as it is too long.
    """
    data = {}
    labels = ['neutral', 'bad', 'good']
    files_by_labels = [files[l] for l in labels]
    p = Pool(3)
    result = p.starmap(clean_reviews_by_label,
                       zip(files_by_labels, labels))
    for i in result:
        data.update(i)
    p.close()
    return data


def clean_reviews_by_label(files, label):
    """
    Clean reviews from service words for one label and add bigrams.
    """
    # word_matrix - list of all words and bigrams of common words
    # all_words - all unique words and all bigrams (most rare will be drop out)
    data = {'word_matrix': [], 'all_words': []}
    # temporary list of all words in all reviews and theit bigrams
    # (final list will be shorter as rare words will be removed)
    allwords = []
    # create tokenizer
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]+')
    n = len(files)
    for i, filepath in enumerate(files):
        if (i+1) % 1000 == 0:
            print('{}: {}/{} docs processed'.format(label, i+1, n))
        # read review and tokenize it
        f = open(filepath)
        bag_words = tokenizer.tokenize(f.read())
        f.close()
        # get part of speech for each word
        lower_words = get_part_of_speech(bag_words)
        # drop service words
        informative_words = choose_informative_words(lower_words)
        # form list of important words by words itself and bigrams
        tokens_bigrams_list = list(
            bigrams(informative_words)) + informative_words
        # add list of words in purpose to calculate frequency by document next
        data['word_matrix'].append(tokens_bigrams_list)
        # add words to big list of all words in all reviews
        allwords.extend(informative_words)
    # find frequencies for all words
    frequencies = FreqDist(allwords)
    # find the least frequent words
    hapaxes = frequencies.hapaxes()
    # remove them
    data['all_words'] = list(set(allwords) - set(hapaxes))
    return {label: data}


def get_part_of_speech(words):
    """
    Get parts of speech for each word in review.
    """
    lower_words = []
    for i in words:
        lower_words.append(i.lower())
    pos_words = pos_tag(lower_words, lang='rus')
    return pos_words


def choose_informative_words(words):
    """
    Leave word if it is noun, verb, adjective or adverb.
    """
    stemmer = SnowballStemmer("russian")
    informative_words = []
    for i in words:
        if i[1] in ['S', 'A', 'V', 'ADV']:
            informative_words.append(stemmer.stem(i[0]))
    return informative_words
