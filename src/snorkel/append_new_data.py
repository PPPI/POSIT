import os
import sys
from multiprocess.pool import Pool

import h5py
import psutil

if sys.platform.startswith('win'):
    import pandas as pd
else:
    import modin.pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import formal_languages
from src.snorkel.classification_based_weak_labelling import classify_labeler_factory
from src.snorkel.snorkel_driver import word2vec_location
from src.snorkel.weak_labellers import *


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def main(argv):
    location = argv[0]
    languages = argv[1:]
    if len(languages) == 0:
        print("You should provide a list of languages to process", file=sys.stderr)

    df_train = pd.read_csv(location)
    print('Loaded data.')

    lfs_tags_per_lang_formal = {
        'uri': [lf_uri_tok],
        'diff': [lf_diff_tok],
        'email': [lf_email_tok],
    }
    if len(languages) == 1 and languages[0].lower() == "language":
        print('Working on Language ID Tagging.')
        # Define the set of labeling functions (LFs)
        lfs_lang = [
            frequency_language_factory(),
            # lang_factory(levenshtein_distance=3),
            lf_builtin_language,
            lf_user_language,
            lf_formal_lang,
        ]
        try:
            h5f = h5py.File('./data/data_votes.h5', 'r')
            L_lang_train_existing = h5f['language_votes'][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()

        applier = PandasLFApplier(lfs_lang)
        L_lang_train = applier.apply(df_train)

        L_lang_train = np.r_[L_lang_train_existing, L_lang_train]

        try:
            os.makedirs('./data', exist_ok=True)
            h5f = h5py.File('./data/data_votes.h5', 'w')
            h5f.create_dataset('language_votes', data=L_lang_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()
    else:
        print('Working on Language Tags.')
        for language in tqdm(languages, desc='Languages'):
            try:
                h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
                L_train_existing = h5f['%s_votes' % language][:]
            finally:
                if 'h5f' in locals().keys():
                    h5f.close()

            if language not in formal_languages:
                clf_labeling_factory = classify_labeler_factory(language, word2vec_location)
                lfs_tags = [x
                            for x in [
                                frequency_labeling_function_factory(language),
                                # frequency_labeling_factories[lang](levenshtein_distance=3),
                                lf_builtin_tag_factory(language),
                            ] +
                            [
                                clf_labeling_factory(n) for n in range(7)
                            ] + [
                                lf_bruteforce_tag_factory(language, tag_encoders)
                            ]
                            if x is not None
                            ]
            else:
                lfs_tags = lfs_tags_per_lang_formal[language]

            tapplier = PandasLFApplier(lfs_tags)
            if sys.platform.startswith('win'):
                L_train = parallelize_dataframe(df_train, tapplier.apply, len(psutil.Process().cpu_affinity()) - 1)
            else:
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
