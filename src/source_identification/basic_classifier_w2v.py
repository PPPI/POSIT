import glob

import numpy as np
from sklearn import preprocessing

from src.source_identification.basic_classifier import run_all_config

try:
    from daal4py.sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    from sklearn.ensemble import RandomForestClassifier


def generate_xy(sources, file_path):
    Xy = list()
    le = preprocessing.LabelEncoder()
    le.fit(sources)
    scaler = preprocessing.RobustScaler()
    for source in sources:
        for file in glob.glob(file_path % source):
            loaded = np.load(file)
            fv = loaded['w2v']
            Xy.append((fv, source))

    X, y = list(zip(*Xy))
    X = np.asarray(X)
    y = le.transform(np.asarray(y))
    scaler.fit(X)

    return X, y, scaler


def main():
    file_path = './data/source_identification/word2vec/%s/feature_vector_*.npz'
    run_all_config(file_path, generate_xy)


if __name__ == '__main__':
    main()
