import os
import sys

from .tagger.data_utils import CorpusIterator
from .tagger_transformers.data_utils import convert_to_iterator_for_transformer
from .tagger_transformers.transformer_config import TransformerConfiguration
from .tagger_transformers.transformer_model import TransformerCoSModel


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

        sentence.strip()

        if sentence == ["exit"]:
            break
        model.tag(sentence, casual=casual)


def main():
    # create instance of config
    config = TransformerConfiguration()
    config.dir_model = sys.argv[1]
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if config.use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create datasets
    test = CorpusIterator(config.filename_test, config.processing_word,
                          config.processing_tag, config.with_l_id, config.max_iter)
    test = convert_to_iterator_for_transformer(test, with_lid=config.with_l_id)

    # build model
    model = TransformerCoSModel(config)
    model.restore_session(config.dir_model)
    # evaluate and interact

    model.validate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
