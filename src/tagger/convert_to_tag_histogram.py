import json
import os
import re
import sys

import numpy as np
from nltk import casual_tokenize

from src.preprocessor.preprocess import CODE_TOKENISATION_REGEX, tokenise_lkml
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel


def process_sent(sentence, casual):
    if casual:
        words_raw = casual_tokenize(sentence.strip())
    else:
        words_raw = [l.strip()
                     for l in re.findall(CODE_TOKENISATION_REGEX,
                                         sentence.strip())
                     if len(l.strip()) > 0]

    return words_raw


def restore_model(config):
    try:
        with open(os.path.join(os.path.dirname(config.dir_model), 'config.json')) as f:
            config.__dict__.update(**json.loads(f.read()))
    except FileNotFoundError:
        pass  # We then need the config file to be correct from the start.

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)
    return model


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    model = restore_model(config)

    tag_vocab = config.vocab_tags
    cooccur = np.zeros(shape=tuple([len(tag_vocab)] * config.nlangs))

    # We preprocess different sources into LKML style then load that
    source = tokenise_lkml(sys.argv[2])
    source_name = sys.argv[3]

    t_feature_vectors = list()
    l_feature_vectors = list()

    for words_raw in source:
        tags_feature_vector = np.zeros(shape=len(tag_vocab))
        lid_feature_vector = np.zeros(shape=config.nlangs)
        tags, lids = model.predict(words_raw)
        for tags, lid in zip(tags, lids):
            for t in tags:
                tags_feature_vector[tag_vocab.token2id(t)] += 1
            cooccur[tuple([tag_vocab.token2id(t) for t in tags])] += 1
            lid_feature_vector[int(lid)] += 1
        t_feature_vectors.append(tags_feature_vector)
        l_feature_vectors.append(lid_feature_vector)

    os.makedirs('./data/source_identification/%s' % source_name, exist_ok=True)
    for idx, (tags, lids) in enumerate(zip(t_feature_vectors, l_feature_vectors)):
        np.savez_compressed(file='./data/source_identification/%s/feature_vector_%d.npz' % (source_name, idx),
                            tags=tags, lids=lids)
    os.makedirs('./data/%s' % source_name, exist_ok=True)
    np.savez_compressed(file='./data/%s/cooccurrence_matrix.npz' % source_name)


if __name__ == "__main__":
    main()
