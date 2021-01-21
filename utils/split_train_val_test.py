"""
Split all reviews in the dataset into three parts:
- train: 78%
- validation: 2%
- test: 20%
"""
import csv
import os
import numpy as np
import argparse

# get from user path to dataset and where to dave result csvs
parser = argparse.ArgumentParser(description='Splitting data into train and test' +
                                             ' for Kinopoisk reviews classification task.')
parser.add_argument('--corpus-root', required=True,
                    help='Path to folder with reviews')
parser.add_argument('--save', default='../data/',
                    help='Path to folder where to save csv files')
args = parser.parse_args()

# get list of labels
labels = os.listdir(args.corpus_root)

# form path to label folders
label_folders = [os.path.join(args.corpus_root, label) for label in labels]
# create path to output dir if doesn't exist
if not os.path.exists(args.save):
    os.makedirs(args.save)

# create writers to output files
train_file = open(os.path.join(args.save, 'train.csv'), 'w')
val_file = open(os.path.join(args.save, 'val.csv'), 'w')
test_file = open(os.path.join(args.save, 'test.csv'), 'w')
train_writer = csv.writer(train_file, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
val_writer = csv.writer(val_file, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
test_writer = csv.writer(test_file, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)

# get revies count for each label
folders_size = [len(os.listdir(f)) for f in label_folders]
# set number of reviews for each label as minimal among all labels
# (actually there are several times more positive reviews than any others,
# but not even distribution among classes will spoil training process)
n_docs_for_one_label = min(folders_size)

for label, folder in zip(labels, label_folders):
    # get necessary amount of reviews randomly
    list_files = np.random.choice(os.listdir(folder),
                                  n_docs_for_one_label,
                                  replace=False)

    files_count = len(list_files)
    # permute files
    list_files_permuted = np.random.permutation(list_files)
    # get number of files for each subset
    train_n = int(0.78 * files_count)
    val_n = int(0.02 * files_count)
    # get files for each subset
    train_files = list_files_permuted[:train_n]
    val_files = list_files_permuted[train_n:train_n + val_n]
    test_files = list_files_permuted[train_n + val_n:]
    # save paths to files from each subset to csv file
    for train_file in train_files:
        train_writer.writerow([label, os.path.join(os.path.join(folder, train_file))])
    for val_file in val_files:
        val_writer.writerow([label, os.path.join(os.path.join(folder, val_file))])
    for test_file in test_files:
        test_writer.writerow([label, os.path.join(os.path.join(folder, test_file))])
