import os
import sys

import h5py
from multiprocess.pool import Pool

if sys.platform.startswith('win'):
    import pandas as pd
else:
    import modin.pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import formal_languages
from src.snorkel.classification_based_weak_labelling_fv import classify_labeler_factory
from src.snorkel.weak_labellers import *


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    results = list(pool.map(func, df_split))
    result = results[0]
    for i in range(1, len(results)):
        result = np.r_[result, results[i]]
    pool.close()
    pool.join()
    return result


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
        except OSError:
            L_lang_train_existing = None
        finally:
            if 'h5f' in locals().keys():
                h5f.close()

        applier = PandasLFApplier(lfs_lang)
        L_lang_train = applier.apply(df_train)

        if L_lang_train_existing is not None:
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
            L_train_existing = None
            try:
                h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
                L_train_existing = h5f['%s_votes' % language][:]
            except OSError:
                L_train_existing = None
            finally:
                if 'h5f' in locals().keys():
                    h5f.close()

            if language not in formal_languages:
                clf_labeling_factory = classify_labeler_factory(language)
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
                L_train = parallelize_dataframe(df_train, tapplier.apply, 6)
            else:
                # modin is not compatible with progress_apply
                L_train = tapplier.apply(df_train, progress_bar=False)

            if L_train_existing is not None:
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
