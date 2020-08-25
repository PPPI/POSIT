import glob
import sys

import numpy as np
from daal4py.sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def main(argv):
    name = 'Random Forest Classifier'
    clf = RandomForestClassifier(max_depth=15, n_estimators=100)
    sources = argv
    Xy = list()
    le = preprocessing.LabelEncoder()
    le.fit(sources)
    for source in sources:
        for file in glob.glob('./data/source_identification/word2vec/%s/feature_vector_*.npz' % source):
            loaded = np.load(file)
            fv = loaded['w2v']
            Xy.append((fv, source))

    X, y = list(zip(*Xy))
    X = np.asarray(X)
    y = le.transform(np.asarray(y))

    split_ratio = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('%s has a mean accuracy of %2.3f' % (name, score))


if __name__ == '__main__':
    main(sys.argv[1:])
