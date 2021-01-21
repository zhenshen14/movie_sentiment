"""
Classify full reviews dataset using 4 models and 2 vectorization types.
May do full analysis for the best model (SVC on TF-IDF).
"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing import prepare_train_data, prepare_test_data
import csv
from time import time


def estimate(model, X_train, Y_train, X_test, Y_test):
    """
    Estimate quality of model on vectorized data.
    Measure time of training and prediction.
    """
    start_time = time()
    # fit train data
    model.fit(X_train, Y_train)
    print("Train time: ", round(time() - start_time, 4))
    start_time = time()
    # predict test data
    Y_predicted = model.predict(X_test)
    print("Test time: ", round(time() - start_time, 4)/len(Y_predicted))
    # get accuracy by comparing predicted and ground-truth labels
    score_test = accuracy_score(Y_test, Y_predicted)
    print("Accuracy: ", round(score_test, 4))


def classify(train_csv, test_csv, vec_type, nb, dectree, svc, mlp):
    """
    classify 4 models for certaing type of vectorization (tf-idf or frequency)
    """
    print("====== {} vector type ======".format(vec_type))
    # load list of paths to reviews
    train_files = get_files_list(train_csv)
    test_files = get_files_list(test_csv)
    # prepare train data (vectorization)
    (X_train, Y_train), important_words, idf = prepare_train_data(
        train_files, vec_type=vec_type)
    print("Train data is prepared.")
    # prepare test data using vocabular of train data
    X_test, Y_test = prepare_test_data(
        test_files, important_words, idf, vec_type=vec_type)
    print("Test data is prepared.")

    print("--- MultinomialNB ---")
    estimate(nb, X_train, Y_train, X_test, Y_test)
    print()

    print("--- DecisionTreeClassifier ---")
    estimate(dectree, X_train, Y_train, X_test, Y_test)
    print()

    print("--- LinearSVC ---")
    estimate(svc, X_train, Y_train, X_test, Y_test)
    print()

    print("--- MLPClassifier ---")
    estimate(mlp, X_train, Y_train, X_test, Y_test)
    print()


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


def analyze_best_model(train_csv, test_csv):
    """
    Estimate precision, recall, accuracy, etc for
    SVC model on tf-idf (per label and in common).
    """
    # define best model
    model = make_pipeline(
        MinMaxScaler(feature_range=(0, 1)),
        LinearSVC(tol=0.001, C=0.01, max_iter=10000,
                  loss='squared_hinge', random_state=1))
    # load list of reviews
    train_files = get_files_list(train_csv)
    test_files = get_files_list(test_csv)
    # prepare train and test data
    (X_train, Y_train), important_words, idf = prepare_train_data(
        train_files, vec_type='tfidf')
    print("Train data is prepared.")
    X_test, Y_test = prepare_test_data(
        test_files, important_words, idf, vec_type='tfidf')
    print("Test data is prepared.")

    # fit on train dataset
    model.fit(X_train, Y_train)
    # predict on test dataset
    Y_predicted = model.predict(X_test)
    # get explicit report for model quality
    report = classification_report(Y_test, Y_predicted)
    print(report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Kinopoisk reviews classification.')
    parser.add_argument('--train', required=True,
                        help='Path to csv file with list of train reviews')
    parser.add_argument('--test', required=True,
                        help='Path to csv file with list of test reviews')
    parser.add_argument('--test-best', action='store_true',
                        help='Path to csv file with list of test reviews')
    args = parser.parse_args()

    # get explicit analysis of the best model
    if args.test_best:
        analyze_best_model(args.train, args.test)
    # estimate quality for all model and vector types (with best parameters)
    else:
        # vectorization with frequency
        nb_freq = MultinomialNB(1.0)
        dectree_freq = DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=1,
            max_depth=30, random_state=1)
        svc_freq = make_pipeline(
            MinMaxScaler(feature_range=(0, 1)),
            LinearSVC(tol=0.001, C=0.0001, max_iter=10000,
                      loss='squared_hinge', random_state=1))
        mlp_freq = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(1000, 100), alpha=0.0001,
                          tol=0.00001, random_state=1))
        classify(args.train, args.test, 'freq',
                 nb_freq, dectree_freq, svc_freq, mlp_freq)

        # vectorization with TF-IDF
        nb_tfidf = MultinomialNB(1.0)
        dectree_tfidf = DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=1,
            max_depth=30, random_state=1)
        svc_tfidf = make_pipeline(
            MinMaxScaler(feature_range=(0, 1)),
            LinearSVC(tol=0.001, C=0.01, max_iter=10000,
                      loss='squared_hinge', random_state=1))
        mlp_tfidf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(2000, 400, 50), alpha=0.00001,
                          tol=0.00001, random_state=1))
        classify(args.train, args.test, 'tfidf',
                 nb_tfidf, dectree_tfidf, svc_tfidf, mlp_tfidf)
