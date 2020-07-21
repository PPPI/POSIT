# Collection of classifiers for consideration
import json

import joblib
import numpy as np
from gensim.corpora import Dictionary
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from snorkel.labeling import labeling_function

from src.preprocessor.codeSearch_preprocessor import languages
from src.tagger.data_utils import camel, snake

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


def train_and_store():
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    for language in languages[:1]:
        # Load Dictionaries so it is consistent with snorkel calls
        tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
        word_dict = Dictionary.load('./data/frequency_data/%s/words.dct' % language)
        # Load Training data
        with open('./data/frequency_data/%s/tag_lookup.json' % language, encoding='utf8') as f:
            tag_lookup = json.loads(f.read())

        # Transform data for training
        X, y = list(), list()
        for tag, examples in tag_lookup.items():
            X += [
                (1 if word[0].isupper() else 0,
                 1 if word[0].istitle() else 0,
                 1 if word[0].islower() else 0,
                 1 if camel(word[0]) else 0,
                 1 if snake(word[0]) else 0,
                 1 if any(char.isupper() for char in word[0][1:]) else 0,
                 1 if any(char.isdigit() for char in word[0]) else 0,
                 1 if set('[~!@#$%^&*()_+{}":;\']+$').intersection(word) else 0,
                 word_dict.token2id[word[0]])
                for word in examples]
            y += [tag_dict.token2id[tag]] * len(examples)

        X = np.asarray(X)
        y = np.asarray(y)

        split_ratio = 0.4
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print('%s has a mean accuracy of %2.3f' % (name, score))
            clf_fname = './data/frequency_data/%s/%s_clf.pkl' % (language, name)
            _ = joblib.dump(clf, clf_fname, compress=9)


def classify_labeler_factory(language):
    classifiers = [joblib.load('./data/frequency_data/%s/%s_clf.pkl' % (language, name)) for name in names]

    word_dict = Dictionary.load('./data/frequency_data/%s/words.dct' % language)

    def classify_using_nth(n):

        @labeling_function()
        def clf_labeler(token):
            try:
                token_id = word_dict[token]
            except KeyError:
                return 0

            return classifiers[n].predict([token_id])[0]

        return clf_labeler

    return classify_using_nth


if __name__ == '__main__':
    train_and_store()
