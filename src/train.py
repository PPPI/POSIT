import json
import os

from .tagger.config import Configuration
from .tagger.data_utils import CorpusIterator
from .tagger.model import CodePoSModel


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # create instance of config
    config = Configuration()
    with open(os.path.join(config.dir_output, 'config.json'), 'w') as f:
        config_out = dict()
        for k in dir(config):
            if k not in ['logger', 'vocab_words', 'vocab_chars', 'vocab_tags', 'processing_word', 'processing_tag',
                         'nwords', 'nchars', 'ntags', 'load']:
                if not (k.startswith('__')):
                    v = getattr(config, k)
                    config_out[k] = v
        f.write(json.dumps(config_out))

    # build model
    model = CodePoSModel(config)
    model.build()

    # create datasets
    dev = CorpusIterator(config.filename_dev, config.processing_word,
                         config.processing_tag, config.with_l_id, config.max_iter)
    train = CorpusIterator(config.filename_train, config.processing_word,
                           config.processing_tag, config.with_l_id, config.max_iter)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
