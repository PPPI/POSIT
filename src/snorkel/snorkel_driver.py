import sys

from gensim.corpora import Dictionary
from snorkel.labeling import LabelModel
from snorkel.labeling import PandasLFApplier

from src.preprocessor.codeSearch_preprocessor import languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]
    # Define the set of labeling functions (LFs)
    lfs_lang = [
        frequency_language_factory(),
        lf_builtin_language,
        lf_uri_lang,
        lf_diff_lang,
        lf_email_lang,
    ]
    lfs_tags = [
                   lf_builtin_tag,
                   lf_uri_tok,
                   lf_diff_tok,
                   lf_email_tok,
               ] + [
                   frequency_labelling_function_factory(language)
                   for language in languages
               ]
    size_tag_voc = 0
    for language in languages:
        tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
        size_tag_voc += len(tag_dict)
    size_lang_voc = 10  # 6 (PL) + 1 (NL) + 3 (FL)

    df_train = SO_to_pandas(location)

    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs_lang)
    L_lang_train = applier.apply(df_train)
    tapplier = PandasLFApplier(lfs_tags)
    L_train = tapplier.apply(df_train)

    # Train the label model and compute the training labels
    lang_label_model = LabelModel(cardinality=size_lang_voc, verbose=True)
    lang_label_model.fit(L_lang_train, n_epochs=2000, log_freq=200, seed=42)
    df_train["lang_label"] = lang_label_model.predict(L=L_lang_train, tie_break_policy="abstain")

    label_model = LabelModel(cardinality=size_tag_voc, verbose=True)
    label_model.fit(L_train, n_epochs=2000, log_freq=200, seed=42)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")

    # TODO: Persist result to disk


if __name__ == '__main__':
    main(sys.argv[1:])
