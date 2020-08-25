import os
import sys

from gensim.models import KeyedVectors

import numpy as np

from src.preprocessor.preprocess import tokenise_lkml
from src.tagger.convert_to_tag_histogram import process_data


def main():
    # We preprocess different sources into LKML style then load that
    word2vec_location = sys.argv[1]
    word2vec_keyedvectors = KeyedVectors.load_word2vec_format(word2vec_location)
    source_name = sys.argv[3]
    if source_name == 'SO':
        source = process_data(sys.argv[2])
    else:
        source = tokenise_lkml(sys.argv[2])

    w_feature_vectors = list()

    for sents_raw in source:
        word2vec_fv = np.zeros_like(word2vec_keyedvectors['I'])
        for words_raw in sents_raw:
            for word in words_raw:
                word2vec_fv += word2vec_keyedvectors[word]
        w_feature_vectors.append(word2vec_fv)

    os.makedirs('./data/source_identification/word2vec/%s' % source_name, exist_ok=True)
    for idx, w2v_fv in enumerate(w_feature_vectors):
        np.savez_compressed(file='./data/source_identification/word2vec/%s/feature_vector_%d.npz' % (source_name, idx),
                            w2v=w2v_fv)


if __name__ == "__main__":
    main()