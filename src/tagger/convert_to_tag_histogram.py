import json
import os
import re
import sys

import numpy as np
from nltk import casual_tokenize, sent_tokenize

from src.preprocessor.preprocess import CODE_TOKENISATION_REGEX, tokenise_lkml
from src.preprocessor.preprocess import tokenize_SO_row
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel


def process_one_rev(target_data, rev, post_id):
    with open(os.path.join(os.path.dirname(target_data), 'SO_Posts', post_id, '%d.html' % rev)) as f:
        html = f.read()
    sents_raw = tokenize_SO_row(html, tag_name='div', all_as_code=True)
    return sents_raw


def process_data(target_data):
    with open(target_data) as f:
        lines_and_revs = [l.strip().split(',') for l in f.readlines()][1:]
    for postId, rev in lines_and_revs:
        rev = int(rev)
        yield [w for w in process_one_rev(target_data, rev, postId) if len(w) > 0]


def process_sent(sentence, casual):
    if casual:
        words_raw = casual_tokenize(sentence.strip())
    else:
        words_raw = [l.strip()
                     for l in re.findall(CODE_TOKENISATION_REGEX,
                                         sentence.strip())
                     if len(l.strip()) > 0]

    return words_raw


def process_docstring(target_data):
    with open(target_data) as f:
        docstrings = f.read().split('<DOC-END>')

    for docstring in docstrings:
        sentences = sent_tokenize(docstring)
        yield [process_sent(sent, False) for sent in sentences]


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

    # We preprocess different sources into LKML style then load that
    source_name = sys.argv[3]
    if source_name == 'SO':
        source = process_data(sys.argv[2])
    elif source_name == 'LKML':
        source = tokenise_lkml(sys.argv[2])
    elif source_name == 'DOCSTRING':
        source = process_docstring(sys.argv[2])

    t_feature_vectors = list()
    l_feature_vectors = list()

    for sents_raw in source:
        tags_feature_vector = np.zeros(shape=len(tag_vocab))
        lid_feature_vector = np.zeros(shape=config.nlangs)
        for words_raw in sents_raw:
            tags, lids = model.predict(words_raw)
            for tags, lid in zip(tags, lids):
                tags_feature_vector[tag_vocab.token2id[tags[model.config.lang_to_id[lid]]]] += 1
                lid_feature_vector[model.config.lang_to_id[lid]] += 1
        t_feature_vectors.append(tags_feature_vector)
        l_feature_vectors.append(lid_feature_vector)

    os.makedirs('./data/source_identification/%s' % source_name, exist_ok=True)
    for idx, (tags, lids) in enumerate(zip(t_feature_vectors, l_feature_vectors)):
        np.savez_compressed(file='./data/source_identification/%s/feature_vector_%d.npz' % (source_name, idx),
                            tags=tags, lids=lids)


if __name__ == "__main__":
    main()
