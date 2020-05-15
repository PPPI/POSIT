import os

from .tagger.data_utils import CorpusIterator
from .tagger_transformers.data_utils import convert_to_iterator_for_transformer
from .tagger_transformers.transformer_config import TransformerConfiguration
from .tagger_transformers.transformer_model import TransformerCoSModel


def main():
    # create instance of config
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    config = TransformerConfiguration()
    if config.use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create datasets
    dev = CorpusIterator(config.filename_dev, config.processing_word,
                         config.processing_tag, config.with_l_id, config.max_iter, 1)
    train = CorpusIterator(config.filename_train, config.processing_word,
                           config.processing_tag, config.with_l_id, config.max_iter, 1)
    train = convert_to_iterator_for_transformer(train, cache=True, with_lid=config.with_l_id)
    dev = convert_to_iterator_for_transformer(dev, with_lid=config.with_l_id)

    # build model
    model = TransformerCoSModel(config)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
