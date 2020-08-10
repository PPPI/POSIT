import os
import sys

import h5py
from gensim.corpora import Dictionary
from nltk import pos_tag, word_tokenize
from snorkel.labeling import LabelModel
from snorkel.labeling import PandasLFApplier
from tqdm import tqdm

from src.preprocessor.codeSearch_preprocessor import languages, natural_languages, formal_languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.classification_based_weak_labelling import classify_labeler_factory
from src.snorkel.encoding import lang_decoding, uri_decoding, diff_decoding, email_decoding
from src.snorkel.weak_labellers import *
from src.tagger.data_utils import O, UNK

tag_decoders = {
    **{
        lang: tag_encoding_factory(lang)[1] for lang in [
            'go',
            'javascript',
            'php',
            'python',
            'ruby',
            'java',
        ]
    },
    **{
        'uri': lambda x: uri_decoding[x] if x > 0 else O,
        'diff': lambda x: diff_decoding[x] if x > 0 else O,
        'email': lambda x: email_decoding[x] if x > 0 else O,
    }
}

word2vec_location = 'G:\\wiki_w2v_models\\wiki-news-300d-1M.vec'  # Update this or move to cli arg


# XXX: Commented out for now as Levenshtein is slow
# lang_factory = frequency_language_levenshtein_factory()
# frequency_labeling_factories = {lang: frequency_labeling_function_levenshtein_factory(lang) for lang in languages}


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

    # Train the label model and compute the training labels
    lang_label_model = LabelModel(cardinality=size_lang_voc, verbose=True)
    lang_label_model.fit(L_lang_train, n_epochs=20000, log_freq=200, seed=42)
    df_train["lang_label"] = lang_label_model.predict(L=L_lang_train, tie_break_policy="random")

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

        # Train the label model and compute the training labels
        label_model = LabelModel(cardinality=size_tag_voc, verbose=True)
        label_model.fit(L_train, n_epochs=20000, log_freq=200, seed=42)
        df_train["label_%s" % language] = label_model.predict(L=L_train, tie_break_policy="random")

    max_post_id = df_train.iloc[-1]['PostIdx']
    valid_index = int(0.6 * max_post_id)
    test_index = int(0.8 * max_post_id)
    os.makedirs('./data/corpora/multilingual/so', exist_ok=True)
    current_context = ''
    for filename in ['eval.txt', 'dev.txt', 'train.txt']:
        with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'w') as f:
            pass
    for index, row in tqdm(df_train.iterrows(), desc='Output'):
        if row['PostIdx'] > max_post_id:
            break

        if row['PostIdx'] > test_index:
            filename = 'eval.txt'
        elif row['PostIdx'] > valid_index:
            filename = 'dev.txt'
        else:
            filename = 'train.txt'
        if row['Context'] != current_context:
            with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'a') as f:
                f.write('\n')
            current_context = row['Context']

        # We use the NLTK pos_tag function for POS tags rather than snorkeling.
        eng_tag = UNK
        for tok, tag in pos_tag(word_tokenize(row['Context'])):
            if tok == str(row['Token']):
                eng_tag = tag
                break

        with open('./data/corpora/multilingual/so/corpus/%s' % filename, 'a') as f:
            to_output = [str(row['Token']), eng_tag] \
                        + [tag_decoders[language](row['label_%s' % language])
                           if row['label_%s' % language] != 0 else O
                           for language in languages + formal_languages] \
                        + [lang_decoding(row['lang_label']) if row['lang_label'] != 0 else row['Language']]
            f.write(' '.join(to_output) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
