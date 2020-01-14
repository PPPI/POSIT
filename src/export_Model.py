import json
import os
import sys

from tensorflow.python.saved_model.simple_save import simple_save

from src.tagger.config import Configuration
from src.tagger.model import CodePoSModel


def main():
    # create instance of config
    config = Configuration()
    config.dir_model = sys.argv[1]
    target_dir = sys.argv[2]

    # build model
    model = CodePoSModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # fully export model
    simple_save(model.sess,
                target_dir,
                inputs=model.get_input_dict(),
                outputs=model.get_output_dict())
    with open(os.path.join(target_dir, 'configuration.json'), 'w') as _:
        json.dumps(config.__dict__)


if __name__ == "__main__":
    main()
