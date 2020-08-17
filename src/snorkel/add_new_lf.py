import os
import sys

import h5py
import numpy as np
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.classification_based_weak_labelling_fv import classify_labeler_factory
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]

    df_train = SO_to_pandas(location)

    for language in tqdm(languages, desc='Languages'):
        try:
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
            L_train_existing = h5f['%s_votes' % language][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()
        clf_labeling_factory = classify_labeler_factory(language)
        lfs_tags = [
                       clf_labeling_factory(n) for n in range(7)
                   ] + [
                       lf_bruteforce_tag_factory(language, tag_encoders)
                   ]
        tapplier = PandasLFApplier(lfs_tags)
        L_train = tapplier.apply(df_train)

        L_train = np.c_[L_train_existing, L_train]

        try:
            os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'w')
            h5f.create_dataset('%s_votes' % language, data=L_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
