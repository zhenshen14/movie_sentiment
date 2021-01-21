"""
Script to choose best parameter for classifier.
Used types of classifier:
- Naive Bayes
- SVM
- Decision Tree
- Multilayer perceptron
"""
from itertools import product
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from classify import get_files_list
from preprocessing import prepare_train_data, prepare_test_data


def choose_multinomial_nb_param(X_train, Y_train, X_test, Y_test):
    """
    Choose parameters for Naive Bayes classifier.
    """
    print('-- Choose for MultinomialNB --')
    # alpha parameter
    params = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
    scores = {}
    for p in params:
        model = MultinomialNB(p)
        model.fit(X_train, Y_train)
        Y_predicted = model.predict(X_test)
        score_test = accuracy_score(Y_test, Y_predicted)
        scores[p] = round(score_test, 4)
    # print results for different parameters
    for alpha, score in scores.items():
        print('alpha {}: {}'.format(alpha, score))
    print()


def choose_svc_param(X_train, Y_train, X_test, Y_test):
    """
    Choose parameter for SVM classifier (with grid search).
    """
    print('-- Choose for LinearSVC --')
    # tolerance
    stop_crit = [0.001, 0.0001]
    # regularization parameter
    reg = [0.0001, 0.001, 0.01, 0.1, 0.5]
    losses = ['squared_hinge', 'hinge']
    for l, st, r in product(losses, stop_crit, reg):
        model = make_pipeline(MinMaxScaler(feature_range=(0, 1)),
                              LinearSVC(
            tol=st, C=r, max_iter=10000, loss=l, random_state=1))
        model.fit(X_train, Y_train)
        Y_predicted = model.predict(X_test)
        score_test = accuracy_score(Y_test, Y_predicted)
        # print results for different parameters
        print('stop: {}, reg: {}, loss: {}    score: {}'.format(
            st, r, l, round(score_test, 4)))
    print()


def choose_decision_tree_param(X_train, Y_train, X_test, Y_test):
    """
    Choose parameter for Decision Tree Classifier (with grid search).
    """
    print('-- Choose for DecisionTreeClassifier --')
    # type of optimization criterium
    crit = ['gini', 'entropy']
    # min samples on decision tree leaf
    min_samples_leaf = [1, 2]
    # max depth of the tree
    max_depth = [30, 50, 100, 200]
    for cr, depth,  min_samples in product(crit, max_depth, min_samples_leaf):
        model = DecisionTreeClassifier(criterion=cr,
                                       min_samples_leaf=min_samples,
                                       max_depth=depth,
                                       random_state=1)
        model.fit(X_train, Y_train)
        Y_predicted = model.predict(X_test)
        score_test = accuracy_score(Y_test, Y_predicted)
        # print results for different parameters
        print('crit: {}, depth: {}, leaf: {},     score: {}'.format(
            cr, depth, min_samples, round(score_test, 4)))
    print()


def choose_neuralnet_param(X_train, Y_train, X_test, Y_test):
    print('-- Choose for MLPClassifier --')
    # define layers count and size
    layers = [(2000, 400), (1000, 100), (2000, 400, 50)]
    # regularization parameter
    alpha = [0.000001, 0.00001, 0.0001]
    for layer, r in product(layers, alpha):
        model = make_pipeline(MinMaxScaler(), MLPClassifier(
            hidden_layer_sizes=layer, alpha=r, tol=0.00001, random_state=1))
        model.fit(X_train, Y_train)
        Y_predicted = model.predict(X_test)
        score_test = accuracy_score(Y_test, Y_predicted)
        # print results for different parameters
        print('layer: {}, alpha: {},   score: {}'.format(
            layer, r, round(score_test, 4)))
    print()


def choose_params(train_files, val_files, vec_type):
    """
    Prepare data according to vectorization type
    and call functions for parameters selection for all types of models.
    """
    print("====== {} vector type ======".format(vec_type))
    (X_train, Y_train), important_words, idf = \
        prepare_train_data(train_files, vec_type=vec_type)
    print("Train data is prepared.")
    X_test, Y_test = \
        prepare_test_data(val_files, important_words, idf, vec_type=vec_type)
    print("Validation data is prepared.")
    print()
    choose_multinomial_nb_param(X_train, Y_train, X_test, Y_test)
    print()
    choose_svc_param(X_train, Y_train, X_test, Y_test)
    print()
    choose_decision_tree_param(X_train, Y_train, X_test, Y_test)
    print()
    choose_neuralnet_param(X_train, Y_train, X_test, Y_test)
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Kinopoisk reviews classification.')
    parser.add_argument('--train', required=True,
                        help='Path to csv file with list of train reviews')
    parser.add_argument('--val', required=True,
                        help='Path to csv file with list of val reviews')
    args = parser.parse_args()

    # load list of reviews
    train_files = get_files_list(args.train, limit=700)
    val_files = get_files_list(args.val, limit=0)

    # choose parameters for data that is vectorizes by frequency
    choose_params(train_files, val_files, 'freq')

    # choose parameters for data that is vectorizes with tf-idf
    choose_params(train_files, val_files, 'tfidf')
