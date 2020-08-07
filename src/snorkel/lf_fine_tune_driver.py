import sys

from gensim.corpora import Dictionary
from snorkel.labeling import LabelModel
from snorkel.labeling import PandasLFApplier

from src.preprocessor.codeSearch_preprocessor import languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]
    language = argv[1]
    df_train = SO_to_pandas(location)

    if language in languages:
        tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
        size_tag_voc = len(tag_dict)
    elif language in ['diff', 'email']:
        size_tag_voc = 2
    else:  # language = 'uri
        size_tag_voc = 7

    lfs_tags = [lf_bruteforce_tag_factory(language, tag_encoders[language])]

    tapplier = PandasLFApplier(lfs_tags)
    L_train = tapplier.apply(df_train)

    label_model = LabelModel(cardinality=size_tag_voc, verbose=True)
    label_model.fit(L_train, n_epochs=20000, log_freq=200, seed=42)
    df_train["label_%s" % language] = label_model.predict(L=L_train, tie_break_policy="random")


if __name__ == '__main__':
    main(sys.argv[1:])
