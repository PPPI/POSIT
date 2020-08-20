import os
import sys

import h5py
import scipy.stats as st
import pandas as pd
from gensim.corpora import Dictionary
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages, natural_languages, formal_languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.classification_based_weak_labelling_fv import classify_labeler_factory
from src.snorkel.weak_labellers import *

if sys.platform.startswith('win'):
    word2vec_location = 'G:\\wiki_w2v_models\\wiki-news-300d-1M.vec'  # Update this or move to cli arg
else:
    word2vec_location = '/mnt/g/wiki_w2v_models/wiki-news-300d-1M.vec'  # Update this or move to cli arg


def main(argv):
    location = argv[0]

    all_languages = languages + natural_languages + formal_languages

    # Not all lf-s exist for all langs, we filter None to avoid issues.
    lfs_tags_per_lang_formal = {
        'uri': [lf_uri_tok],
        'diff': [lf_diff_tok],
        'email': [lf_email_tok],
    }
    size_lang_voc = len(all_languages)  # 6 (PL) + 1 (NL) + 3 (FL) = 10

    df_train = SO_to_pandas(location)

    # Apply the LFs to the unlabeled training data
    try:
        try:
            h5f = h5py.File('./data/data_votes.h5', 'r')
            L_lang_train = h5f['language_votes'][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()
    except (OSError, FileNotFoundError, KeyError):
        # Define the set of labeling functions (LFs)
        lfs_lang = [
            frequency_language_factory(),
            # lang_factory(levenshtein_distance=3),
            lf_builtin_language,
            lf_user_language,
            lf_formal_lang,
        ]
        applier = PandasLFApplier(lfs_lang)
        L_lang_train = applier.apply(df_train)
        try:
            os.makedirs('./data', exist_ok=True)
            h5f = h5py.File('./data/data_votes.h5', 'w')
            h5f.create_dataset('language_votes', data=L_lang_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()

    # Take the mode and ignore the counts array
    L_lang_train = st.mode(L_lang_train, axis=1, nan_policy='omit')[0]
    df_train["lang_label"] = pd.Series(L_lang_train)

    for language in tqdm(languages + formal_languages, desc='Languages'):
        if language in languages:
            tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
            size_tag_voc = len(tag_dict)
        elif language in ['diff', 'email']:
            size_tag_voc = 2
        else:  # language = 'uri
            size_tag_voc = 7

        try:
            try:
                h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
                L_train = h5f['%s_votes' % language][:]
            finally:
                if 'h5f' in locals().keys():
                    h5f.close()
        except (OSError, FileNotFoundError, KeyError):
            if language in languages:
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
            # Apply the LFs to the unlabeled training data
            tapplier = PandasLFApplier(lfs_tags)
            L_train = tapplier.apply(df_train)
            try:
                os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
                h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'w')
                h5f.create_dataset('%s_votes' % language, data=L_train)
            finally:
                if 'h5f' in locals().keys():
                    h5f.close()

        # Take the mode and ignore the counts array
        L_train = st.mode(L_train, axis=1, nan_policy='omit')[0]
        df_train["label_%s" % language] = pd.Series(L_train)

    df_train.to_csv(location[:-len('.csv')] + '_annotated_majority.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
