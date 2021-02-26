import sys

import nltk
import spacy
from sklearn.metrics import cohen_kappa_score

from src.preprocessor.preprocess import tokenise_SO, tokenise_lkml
from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel


def process_data(model, target_data, stackoverflow=False):
    if stackoverflow:
        source = tokenise_SO(target_data, 75000, 76000, True)
    else:
        source = tokenise_lkml(target_data)

    sents = list()
    for sents_raw, _ in source:
        for words_raw in sents_raw:
            if len(words_raw) > 0:
                preds = model.predict(words_raw)
                eng_tags = preds[0][0]
                sents.append(list(zip(words_raw, eng_tags)))

    return sents


def process_nltk(target_data, stackoverflow=False):
    if stackoverflow:
        source = tokenise_SO(target_data, 75000, 76000, True)
    else:
        source = tokenise_lkml(target_data)

    sents = list()
    for sents_raw, _ in source:
        for words_raw in sents_raw:
            sents.append(nltk.pos_tag(words_raw))

    return sents


def process_spacy(target_data, stackoverflow=False):
    nlp = spacy.load("en_core_web_trf")
    if stackoverflow:
        source = tokenise_SO(target_data, 75000, 76000, True)
    else:
        source = tokenise_lkml(target_data)

    sents = list()
    for sents_raw, _ in source:
        for words_raw in sents_raw:
            words = nlp(' '.join(words_raw))
            sents.append([(tok.text, tok.pos_) for tok in words])

    return sents


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    target_data = sys.argv[2]
    stackoverflow = sys.argv[3].lower() == 'true'

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # run model over given data
    posit_tagging = process_data(model, target_data, stackoverflow=stackoverflow)
    nltk_tagging = process_nltk(target_data, stackoverflow=stackoverflow)
    spacy_tagging = process_spacy(target_data, stackoverflow=stackoverflow)

    for first, second in [(posit_tagging, nltk_tagging), (posit_tagging, spacy_tagging), (nltk_tagging, spacy_tagging)]:
        print("%2.3f" % cohen_kappa_score([p[-1] for p in first], [p[-1] for p in second]))


if __name__ == "__main__":
    main()
