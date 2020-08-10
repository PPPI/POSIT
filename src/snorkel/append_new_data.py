import os
import sys

import h5py
import numpy as np
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages, formal_languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.classification_based_weak_labelling import classify_labeler_factory
from src.snorkel.snorkel_driver import word2vec_location
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]

    df_train = SO_to_pandas(location)

    lfs_tags_per_lang_formal = {
        'uri': [lf_uri_tok],
        'diff': [lf_diff_tok],
        'email': [lf_email_tok],
    }

    for language in languages:
        try:
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
            L_train_existing = h5f['%s_votes' % language][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()

        if language in tqdm(languages + formal_languages, desc='Languages'):
            clf_labeling_factory = classify_labeler_factory(language, word2vec_location)
            lfs_tags = [x for x in [
                frequency_labeling_function_factory(language),
                # frequency_labeling_factories[lang](levenshtein_distance=3),
                lf_builtin_tag_factory(language),
                lf_bruteforce_tag_factory(language, tag_encoders)
            ] +
                        [clf_labeling_factory(n) for n in range(7)]
                        if x is not None]
        else:
            lfs_tags = lfs_tags_per_lang_formal[language]

        tapplier = PandasLFApplier(lfs_tags)
        L_train = tapplier.apply(df_train)

        L_train = np.r_[L_train_existing, L_train]

        try:
            os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'w')
            h5f.create_dataset('%s_votes' % language, data=L_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
