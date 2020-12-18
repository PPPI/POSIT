import glob
from itertools import chain, combinations

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

try:
    from daal4py.sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    from sklearn.ensemble import RandomForestClassifier


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


def generate_all_configs():
    sources_cat = {
        'SO': [['SO'], [f"SO-{l}" for l in ['java', 'javascript', 'python', 'go', 'ruby', 'php']]],
        'DOCSTRING': [['DOCSTRING'], [f"DOCSTRING-{l}" for l in ['java', 'javascript', 'python', 'go', 'ruby', 'php']]],
        'LKML': [['LKML']],
    }

    configs = list()
    for so_s in sources_cat['SO']:
        for doc_s in sources_cat['DOCSTRING']:
            for email_s in sources_cat['LKML']:
                for so_sub in all_subsets(so_s):
                    for doc_sub in all_subsets(doc_s):
                        for email_sub in all_subsets(email_s):
                            config = so_sub + doc_sub + email_sub
                            if len(config) > 1:
                                configs.append(config)

    return configs


def generate_xy(sources, file_path):
    Xy = list()
    le = preprocessing.LabelEncoder()
    le.fit(sources)
    scaler = preprocessing.RobustScaler()
    for source in sources:
        for file in glob.glob(file_path % source):
            loaded = np.load(file)
            tags = loaded['tags']
            lids = loaded['lids']
            fv = np.concatenate((tags, lids))
            Xy.append((fv, source))

    X, y = list(zip(*Xy))
    X = np.asarray(X)
    y = le.transform(np.asarray(y))
    scaler.fit(X)

    return X, y, scaler


def run_eval(clf, scaler, X, y):
    kf = StratifiedKFold(10)
    scores = list()
    for k, (train, test) in enumerate(kf.split(X, y)):
        try:
            clf.fit(scaler.transform(X[train]), y[train])
            score = clf.score(scaler.transform(X[test]), y[test])
            scores.append(score)
        except ValueError:
            pass

    return scores


def run_all_config(file_path, generate_xy=generate_xy, suffix='posit'):
    results = {'Method': list(), 'Mean': list(), 'STD': list()}
    configs = generate_all_configs()
    for sources in configs:
        print(f"Solving for {sources}")
        name = 'Random Forest Classifier'
        clf = RandomForestClassifier(n_estimators=100)
        X, y, scaler = generate_xy(sources, file_path)
        scores = run_eval(clf, scaler, X, y)
        print(f"{name} has a mean accuracy of {np.mean(scores)}+-{1.6449 * np.std(scores)}")
        results['Method'].append(str(sources))
        results['Mean'].append(np.mean(scores))
        results['STD'].append(np.std(scores))

    df = pd.DataFrame(data=results)
    df.to_csv(f"./results/source_separation_{suffix}.csv", index_label='Method')


def main():
    file_path = './data/source_identification/%s/feature_vector_*.npz'
    run_all_config(file_path)


if __name__ == '__main__':
    main()
