import os
import sys

import h5py
from gensim.corpora import Dictionary
from snorkel.labeling import PandasLFApplier

from src.preprocessor.codeSearch_preprocessor import languages
from src.preprocessor.so_to_pandas import SO_to_pandas
from src.snorkel.weak_labellers import *


def main(argv):
    location = argv[0]

    df_train = SO_to_pandas(location)

    for language in languages:
        tag_dict = Dictionary.load('./data/frequency_data/%s/tags.dct' % language)
        size_tag_voc = len(tag_dict)
        try:
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'r')
            L_train_existing = h5f['%s_votes' % language][:]
        finally:
            if 'h5f' in locals().keys():
                h5f.close()

        lfs_tags = [lf_bruteforce_tag_factory(language, tag_encoders)]
        tapplier = PandasLFApplier(lfs_tags)
        L_train = tapplier.apply(df_train)

        # TODO: np.stack/merge L_train_existing and L_train

        try:
            os.makedirs('./data/frequency_data/%s' % language, exist_ok=True)
            h5f = h5py.File('./data/frequency_data/%s/data_votes.h5' % language, 'w')
            h5f.create_dataset('%s_votes' % language, data=L_train)
        finally:
            if 'h5f' in locals().keys():
                h5f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
