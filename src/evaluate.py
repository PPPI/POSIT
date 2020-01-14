import json
import os
import re
import sys

from nltk import casual_tokenize

from .preprocessor.preprocess import CODE_TOKENISATION_REGEX
from .tagger.config import Configuration
from .tagger.data_utils import CorpusIterator
from .tagger.model import CodePoSModel


def interactive_shell(model, casual=False):
    """Creates interactive shell to play with model
    Args:
        model: instance of NERModel
        casual: If we should use the nltk casual tokenize
    """
    model.logger.info("""
        This is an interactive mode.
        To exit, enter 'exit'.
        You can enter a sentence like
        input> If you have a java.io.InputStream object, how should you process that object and produce a String?""")

    while True:
        sentence = input("input> ")

        if casual:
            words_raw = casual_tokenize(sentence.strip())
        else:
            words_raw = [l.strip()
                         for l in re.findall(CODE_TOKENISATION_REGEX,
                                             sentence.strip())
                         if len(l.strip()) > 0]

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        if isinstance(preds, tuple):
            preds = preds[0]

        print(' '.join(['%s_%s' % (w, t) for w, t in zip(words_raw, preds)]))


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

    # create dataset
    test = CorpusIterator(config.filename_test, config.processing_word,
                          config.processing_tag, config.with_l_id, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
