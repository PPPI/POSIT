import os
import uuid

from gensim.corpora import Dictionary

from .data_utils import get_processing_word
from .general_utils import get_logger


class Configuration:
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

    # noinspection PyTypeChecker
    def load(self):
        # 1. vocabulary
        self.vocab_words = Dictionary.load(self.filename_words)
        self.vocab_tags = Dictionary.load(self.filename_tags)
        self.vocab_chars = Dictionary.load(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=False, chars=self.use_chars,
                                                   feature_vector=self.use_features)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False)

    # corpus specific hints
    multilang = True
    nlangs = 10  # The number of Natural and Formal (including Programming) languages
    if multilang:
        id_to_lang = {n: l for n, l in
                      enumerate(
                          ['English', 'go', 'java', 'javascript', 'php', 'python', 'ruby', 'uri', 'email', 'diff'])}
        lang_to_id = {v: k for k, v in id_to_lang.items()}
        # This is to allow accuracy for just PL languages, 0 is Eng, 7,8,9 are formals that are non-PL
        non_pl_lang_ids = [0, 7, 8, 9]
    else:
        id_to_lang = None
        lang_to_id = None
        # This is to allow accuracy for just PL languages, 0 is Eng
        non_pl_lang_ids = [0]

    # general config
    with_l_id = True
    with_l_id = with_l_id or multilang
    project = "so_majority"
    project += '_Id' if with_l_id and not multilang else ''
    # project += '5'

    # embeddings
    dim_word = 300
    dim_char = 100
    embeddings = None

    # dataset
    filename_dev = "data/corpora/%s/corpus/dev.txt" % ('multilingual/%s' % project if multilang else project)
    filename_test = "data/corpora/%s/corpus/eval.txt" % ('multilingual/%s' % project if multilang else project)
    filename_train = "data/corpora/%s/corpus/train.txt" % ('multilingual/%s' % project if multilang else project)

    max_iter = None  # 20000  # if not None, max number of examples in dataset

    # vocab
    filename_words = "data/corpora/%s/words.dct" % ('multilingual/%s' % project if multilang else project)
    filename_tags = "data/corpora/%s/tags.dct" % ('multilingual/%s' % project if multilang else project)
    filename_chars = "data/corpora/%s/chars.dct" % ('multilingual/%s' % project if multilang else project)

    # training
    train_embeddings = True
    nr_epochs = 30
    dropout = 0.5
    batch_size = 4
     
    lr_method = "adam"
    lr = 0.005
    lr_decay = 0.90
    clip = None  # if None, no clipping
    nr_epochs_no_imprvmt = 3

    # model hyperparameters
    max_len = 100  # The maximum sentence/snippet length we train on
    n_features = 8  # This is predefined by design, should be the size of our feature vector
    hidden_size_char = 16  # lstm on chars
    hidden_size_features = 8  # lstm on feature vector
    hidden_size_lstm = 48  # lstm on word embeddings
    if with_l_id:
        if project == 'SO_Freq_Id':
            class_weight = 1 - 0.144  # For SO_Freq_Id it is: 740438 / (740438 + 4394836)
        else:
            class_weight = 0.5
        l_id_weight = 0.5
        # Hyper-params for MLP going from state to bi-LSTM output to L_ID
        n_hidden_1 = 128  # 1st layer number of neurons
        n_hidden_2 = 32  # 2nd layer number of neurons
        if multilang:
            n_lang = nlangs
        else:
            n_lang = 2  # Number of languages being mixed

    use_cpu = False
    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = False  # if char embedding, training is 3.5x slower on CPU
    use_features = False

    # In batch shuffle of training examples
    seed = 42
    shuffle = True

    # storage config
    config_str = ''
    if use_crf:
        config_str += '_with_crf'
    else:
        config_str += '_without_crf'
    if use_chars:
        config_str += '_with_chars'
    else:
        config_str += '_without_chars'
    if use_features:
        config_str += '_with_features'
    else:
        config_str += '_without_features'
    config_str += '_epochs_%d_dropout_%2.3f_batch_%d_opt_%s_lr_%2.4f_lrdecay_%2.4f' \
                  % (nr_epochs, dropout, batch_size, lr_method, lr, lr_decay)
    dir_output = "results/test/%s/%s/" % (('multilingual/%s' % project if multilang else project),
                                          str(uuid.uuid4()) + config_str)
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"
