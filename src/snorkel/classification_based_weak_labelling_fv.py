import json

import joblib
import numpy as np
from gensim.corpora import Dictionary
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from snorkel.labeling import labeling_function
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages
from src.tagger.data_utils import camel, snake

names = [
    "Nearest Neighbors",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "Linear SVM",
    # "RBF SVM",
]

np.random.seed(42)


def to_feature_vector(word):
    return [1 if word.isupper() else 0,
            1 if word.istitle() else 0,
            1 if word.islower() else 0,
            1 if camel(word) else 0,
            1 if snake(word) else 0,
            1 if any(char.isupper() for char in word[1:]) else 0,
            1 if any(char.isdigit() for char in word) else 0,
            1 if set('[~!@#$%^&*()_+{}":;\']+$').intersection(word) else 0]


def train_and_store():
    classifiers = [
        KNeighborsClassifier(5),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),  # Not enough RAM to run this locally
        DecisionTreeClassifier(max_depth=15),
        RandomForestClassifier(max_depth=15, n_estimators=100),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),  # Time out on 5h waiting
    ]

    for language in tqdm(languages):
        print('Working on %s' % language)
        # Load Dictionaries so it is consistent with snorkel calls
        tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
        # Load Training data
        with open('./data/frequency_data/%s/tag_lookup.json' % language, encoding='utf8') as f:
            tag_lookup = json.loads(f.read())

        # Transform data for training
        Xy = list()
        for tag, examples in tag_lookup.items():
            np.random.shuffle(examples)
            examples = examples[:]
            Xy += [
                (to_feature_vector(word[0]), tag_dict.token2id[tag])
                for word in examples
            ]

        X, y = list(zip(*Xy))
        X = np.asarray(X)
        y = np.asarray(y)

        split_ratio = 0.4
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print('%s has a mean accuracy of %2.3f' % (name, score))
            clf_fname = './data/frequency_data/%s/%s_clf_fv.pkl' % (language, name)
            _ = joblib.dump(clf, clf_fname, compress=9)


def classify_labeler_factory(language):
    classifiers = [joblib.load('./data/frequency_data/%s/%s_clf_fv.pkl' % (language, name)) for name in names]

    def classify_using_nth(n):
        @labeling_function(name='clf_labeler_%d' % n)
        def clf_labeler(token):
            feature_vector = to_feature_vector(token)

            return classifiers[n].predict([feature_vector])[0]

        return clf_labeler

    return classify_using_nth


if __name__ == '__main__':
    train_and_store()
