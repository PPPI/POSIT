import json
from multiprocessing.pool import Pool

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
from src.snorkel.weak_labellers import ABSTAIN
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


def process_language(language):
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

    for name, clf in tqdm(zip(names, classifiers), total=len(classifiers), desc='Classifiers for %s' % language):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('%s has a mean accuracy of %2.3f' % (name, score))
        clf_fname = './data/frequency_data/%s/%s_clf_fv.pkl' % (language, name)
        _ = joblib.dump(clf, clf_fname, compress=9)


def train_and_store():
    n_cores = 6

    pool = Pool(n_cores)
    pool.map(process_language, languages)
    pool.close()
    pool.join()


def classify_labeler_factory(language):
    classifiers = list()
    for name in names:
        try:
            classifiers.append(joblib.load('./data/frequency_data/%s/%s_clf_fv.pkl' % (language, name)))
        except FileNotFoundError:
            classifiers.append(None)

    def classify_using_nth(n):
        @labeling_function(name='clf_labeler_%d' % n)
        def clf_labeler(row):
            feature_vector = to_feature_vector(str(row['Token']))

            if classifiers[n] is not None:
                return classifiers[n].predict([feature_vector])[0]
            else:
                return ABSTAIN

        return clf_labeler

    return classify_using_nth


if __name__ == '__main__':
    train_and_store()