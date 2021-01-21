from nltk.probability import FreqDist
from preprocessing import clean_reviews, get_important_words
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from scipy.sparse import csr_matrix


def get_files_list(csv_file, limit=0):
    """
    Get list of paths to reviews from csv.
    """
    # may load only subset of data
    limit = limit if limit > 0 else 100000
    # split by labels
    labels = ['neutral', 'bad', 'good']
    files = {l: [] for l in labels}
    with open(csv_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='|')
        for row in csvreader:
            files[row[0]].append(row[1])
    # form dictionary label - list of reviews
    files = {l: f[:limit] for l, f in files.items()}
    return files


train_files = get_files_list('data/train.csv')
val_files = get_files_list('data/val.csv')
test_files = get_files_list('data/test.csv')

labels = ['neutral', 'bad', 'good']
files = {l: [] for l in labels}
for l in [train_files, val_files, test_files]:
    for label in files.keys():
        files[label].extend(l[label][:1000])
"""
lengths = []
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]+')
for labelname, l in files.items():
    for filepath in tqdm(l):
        f = open(filepath)
        bag_words = tokenizer.tokenize(f.read())
        f.close()
        lengths.append(len(bag_words))

plt.hist(lengths, bins=100)
plt.xlabel('Length of review')
plt.ylabel('Count of reviews')
plt.title('Distribution of review\'s length')
plt.savefig('length_distribution.png')
"""


data = clean_reviews(files)
# get 10000 most frequent words for each label
important_words = get_important_words(data)


def get_freq_vector(data, important_words, label):
    """
    Vectorize data by frequency.
    """
    # create list of pairs of type:
    # [(list_of_review_words, class_label)]
    labeled_data = []
    for document in data[label]['word_matrix']:
        labeled_data.append((document, label))

    # prepare dense matrix
    matrix_vec = np.zeros((len(important_words)), dtype=np.uint32)
    for index_doc, document in enumerate(tqdm(labeled_data)):
        for index_word, word in enumerate(important_words):
            # count frequency inside one review
            matrix_vec[index_word] += document[0].count(word)
    return matrix_vec


most_freq_words = []
good_freq = get_freq_vector(data, important_words, 'good')
neutral_freq = get_freq_vector(data, important_words, 'neutral')
bad_freq = get_freq_vector(data, important_words, 'bad')
total_freq = good_freq+neutral_freq+bad_freq
most_freq_words = total_freq.argsort()[-14:][::-1]

freq = {important_words[i]: [good_freq[i], neutral_freq[i], bad_freq[i]] for i in most_freq_words}

labels = freq.keys()

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

good = [i[0] for i in freq.values()]
neutral = [i[1] for i in freq.values()]
bad = [i[2] for i in freq.values()]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, good, width, label='Good')
rects2 = ax.bar(x, neutral, width, label='Neutral')
rects3 = ax.bar(x + width, bad, width, label='Bad')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency')
ax.set_title('Most frequent words')
ax.set_xticks(x)
ax.tick_params(axis='x', rotation=60)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()


# freqs = {}
# for label in labels:
#     allwords = data[label]['all_words']
#     freqs[label] = FreqDist(allwords)

# list_words = {}
# for label in labels:
#     important = freqs[label].most_common(5)
#     print(important)
#     for i in important:
#         list_words[i[0]] = []
#         for l in labels:
#             list_words[i[0]].append(freqs[l].freq(i[0]))
# print(list_words)
