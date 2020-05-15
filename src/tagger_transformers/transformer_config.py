import os
import uuid

import tensorflow as tf
from gensim.corpora import Dictionary

from ..tagger.data_utils import get_processing_word
from ..tagger.general_utils import get_logger


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TransformerConfiguration:
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        :param load: (bool) if True, load embeddings into np array, else None
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        # 1. vocabulary
        self.vocab_words = Dictionary.load(self.filename_words)
        self.vocab_tags = Dictionary.load(self.filename_tags)

        self.nwords = len(self.vocab_words) + 1
        self.ntags = len(self.vocab_tags) + 1

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, 
                                                   lowercase=False, 
                                                   chars=False,
                                                   feature_vector=False, offset=1)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False,
                                                  offset=1)

    # general config
    with_l_id = True
    project = "SO_n_Lucid"
    project += '_Id' if with_l_id else ''
    dir_output = "results/test/%s/%s/" % (project, str(uuid.uuid4()))
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    # dataset
    filename_dev = "data/corpora/%s/corpus/dev.txt" % project
    filename_test = "data/corpora/%s/corpus/eval.txt" % project
    filename_train = "data/corpora/%s/corpus/train.txt" % project

    max_iter = None  # if not None, max number of examples in dataset

    # vocab
    filename_words = "data/corpora/%s/words.dct" % project
    filename_tags = "data/corpora/%s/tags.dct" % project
    filename_chars = "data/corpora/%s/chars.dct" % project

    # training
    batch_size = 1280
    dropout_rate = 0.1
    nr_epochs = 1000
    nr_epochs_no_imprvmt = 5

    # model hyperparameters
    num_layers = 2
    d_model = 256
    dff = 256
    num_heads = 8
    max_len = 48
    lr = CustomSchedule(d_model)

    use_cpu = False

    # In batch shuffle of training examples
    buffer_size = 8192
    seed = 42
