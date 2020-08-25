import glob
import sys

import numpy as np
from daal4py.sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold


def main(argv):
    name = 'Random Forest Classifier'
    clf = RandomForestClassifier(n_estimators=100)
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

    kf = KFold(10)
    scores = list()
    for k, (train, test) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        score = clf.score(X[test], y[test])
        scores.append(score)
    print('%s has a mean accuracy of %2.3f' % (name, np.mean(scores)))


if __name__ == '__main__':
    main(sys.argv[1:])
