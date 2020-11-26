import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from .base_model import BaseModel
from .data_utils import pad_sequences, minibatches
from .general_utils import Progbar


class CodePoSModel(BaseModel):
    """Specialized class of Model for PoS tagging"""

    def __init__(self, config):
        super(CodePoSModel, self).__init__(config)
        self.use_cpu = config.use_cpu
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.token2id.items()}
        self.max_len = config.max_len

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                 name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None],
                                                         name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None],
                                                 name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                     name="word_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.feature_vector = tf.compat.v1.placeholder(tf.int32, shape=[None, None, self.config.n_features],
                                                       name="feature_vector")

        # shape = (batch_size, max_length of sentence)
        self.feature_sizes = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                      name="feature_sizes")

        if self.config.multilang:
            # shape = (batch size, max length of sentence in batch, number of languages)
            self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None],
                                                   name="labels")
        else:
            # shape = (batch size, max length of sentence in batch)
            self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                   name="labels")

        if self.config.with_l_id:
            # shape = (batch size, max length of sentence in batch)
            self.labels_l = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                     name="labels_l")

        # hyper parameters
        self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                                name="dropout")
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                           name="lr")

    def get_feed_dict(self, words, labels=None, labels_l=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            labels_l: list of language ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # Use labels as proxy for training, only reduce to max_len in training
        if labels is not None:
            words = words[:self.max_len]
        if labels is not None:
            labels = labels[:self.max_len]
        if labels_l is not None:
            labels_l = labels_l[:self.max_len]
        # perform padding of the given data
        if self.config.use_chars:
            if self.config.use_features:
                features, char_ids, word_ids = zip(*words)
                feature_vectors, feature_lengths = pad_sequences(features, pad_tok=0, nlevels=2)
                word_ids, sequence_lengths = pad_sequences(word_ids, 0)
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            else:
                char_ids, word_ids = zip(*words)
                word_ids, sequence_lengths = pad_sequences(word_ids, 0)
                char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            if self.config.use_features:
                features, word_ids = zip(*words)
                feature_vectors, feature_lengths = pad_sequences(features, pad_tok=0, nlevels=2)
                word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            else:
                word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths,
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if self.config.use_features:
            feed[self.feature_vector] = feature_vectors
            feed[self.feature_sizes] = feature_lengths

        if labels is not None:
            if self.config.multilang:
                labels, _ = pad_sequences(labels, pad_tok=0, nlevels=2)
            else:
                labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if self.config.with_l_id:
            if labels_l is not None:
                labels_l, _ = pad_sequences(labels_l, 0)
                feed[self.labels_l] = labels_l

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def get_input_dict(self, ):
        """Given a model, create a record of the input variables
        Returns:
            dict {name: placeholder_tensor}

        """
        # build feed dictionary
        input_dict = {
            'word_ids': self.word_ids,
            'sequence_lengths': self.sequence_lengths,
        }

        if self.config.use_chars:
            input_dict['char_ids'] = self.char_ids
            input_dict['word_lengths'] = self.word_lengths

        if self.config.use_features:
            input_dict['feature_vectors'] = self.feature_vector
            input_dict['feature_lengths'] = self.feature_sizes

        return input_dict

    def get_output_dict(self, ):
        """Given a model, create a record of the output variables
        Returns:
            dict {name: placeholder_tensor}

        """
        output_dict = {'labels': self.labels}

        if self.config.with_l_id:
            output_dict['labels_l'] = self.labels_l

        return output_dict

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.compat.v1.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.compat.v1.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.compat.v1.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.compat.v1.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.keras.layers.LSTMCell(self.config.hidden_size_char)
                cell_bw = tf.keras.layers.LSTMCell(self.config.hidden_size_char)
                _output = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        with tf.compat.v1.variable_scope("feature"):
            if self.config.use_features:
                s = tf.shape(self.feature_vector)
                if self.use_cpu:
                    feature_vector = tf.reshape(self.feature_vector, shape=[s[0] * s[1], self.config.n_features, 1])
                else:
                    feature_vector = tf.reshape(self.feature_vector, shape=[s[0] * s[1], 1, self.config.n_features])
                feature_sizes = tf.reshape(self.feature_sizes, shape=[s[0] * s[1]])

                cell_feature_fw = tf.keras.layers.LSTMCell(self.config.hidden_size_features)
                cell_feature_bw = tf.keras.layers.LSTMCell(self.config.hidden_size_features)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_feature_fw, cell_feature_bw, tf.cast(feature_vector, tf.float32),
                    sequence_length=feature_sizes, dtype=tf.float32)

                # read and concat output
                _, ((_, output_feature_fw), (_, output_feature_bw)) = _output
                output_feature = tf.concat([output_feature_fw, output_feature_bw], axis=-1)

                # shape = (batch size, max sentence length, feature hidden size)
                output_feature = tf.reshape(output_feature,
                                            shape=[s[0], s[1], 2 * self.config.hidden_size_features])
                word_embeddings = tf.concat([word_embeddings, output_feature], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, rate=1 - self.dropout)

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.compat.v1.variable_scope("bi-lstm"):
            # Use implementation = 1 to force more ops of smaller size (to fit our GPU)
            cell_fw = tf.keras.layers.LSTMCell(self.config.hidden_size_lstm,
                                               implementation=1,
                                               recurrent_dropout=self.config.dropout)
            cell_bw = tf.keras.layers.LSTMCell(self.config.hidden_size_lstm,
                                               implementation=1,
                                               recurrent_dropout=self.config.dropout)
            (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, rate=1 - self.dropout)

        with tf.compat.v1.variable_scope("proj"):
            if self.config.multilang:
                W = tf.compat.v1.get_variable("W", dtype=tf.float32,
                                              shape=[2 * self.config.hidden_size_lstm,
                                                     self.config.nlangs * self.config.ntags])

                b = tf.compat.v1.get_variable("b", shape=[self.config.nlangs * self.config.ntags],
                                              dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                W = tf.compat.v1.get_variable("W", dtype=tf.float32,
                                              shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

                b = tf.compat.v1.get_variable("b", shape=[self.config.ntags],
                                              dtype=tf.float32, initializer=tf.zeros_initializer())

            self.nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            # This should be [batch, steps, nlangs,  ntags] and propagate back from here!
            if self.config.multilang:
                self.logits = tf.reshape(pred, [-1, self.nsteps, self.config.nlangs, self.config.ntags])
            else:
                self.logits = tf.reshape(pred, [-1, self.nsteps, self.config.ntags])

            if self.config.with_l_id:
                # Store layers weight & bias
                weights = {
                    'h1': tf.Variable(tf.random.normal([2 * self.config.hidden_size_lstm, self.config.n_hidden_1])),
                    'h2': tf.Variable(tf.random.normal([self.config.n_hidden_1, self.config.n_hidden_2])),
                    'out': tf.Variable(tf.random.normal([self.config.n_hidden_2, self.config.n_lang]))
                }
                biases = {
                    'b1': tf.Variable(tf.random.normal([self.config.n_hidden_1])),  # , name='b1'),
                    'b2': tf.Variable(tf.random.normal([self.config.n_hidden_2])),  # , name='b2'),
                    'out': tf.Variable(tf.random.normal([self.config.n_lang])),  # , name='bout')
                }

                layer_1 = tf.tanh(tf.add(tf.matmul(output, weights['h1']), biases['b1']))

                layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

                self.logits_l = tf.reshape(out_layer, [-1, self.nsteps, self.config.n_lang])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With the CRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)
            if self.config.with_l_id:
                self.labels_pred_l = tf.cast(tf.argmax(self.logits_l, axis=-1),
                                             tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        # NOTE: Masking the null_class (O) makes the system over-predict raw_identifier, should we somehow balance the
        # weights of labels by their occurrence count?
        # Note from a future Profir: Balancing code was written with this comment, use git blame if needed
        # to do the same for per label in the multiclass prediction case.
        if self.config.use_crf:
            if self.config.multilang:
                log_likelihood = 0
                self.trans_params = list()
                for dim in range(self.config.nlangs):  # One prediction per language + language prediction stays in l_id
                    self.current_logits = tf.reshape(self.logits[:, :, dim:dim + 1, :],
                                                     shape=[-1, self.nsteps, self.config.ntags],
                                                     )
                    self.current_labels = tf.reshape(self.labels[:, :, dim:dim + 1],
                                                     shape=[-1, self.nsteps],
                                                     )
                    with tf.compat.v1.variable_scope("Language_%d" % dim):
                        current_log_likelihood, trans_params = tfa.text.crf.crf_log_likelihood(
                            self.current_logits, self.current_labels, self.sequence_lengths)
                        self.trans_params.append(trans_params)
                        # Only propagate for the languages that exist in the current batch
                        log_likelihood += tf.multiply(tf.cast(
                            tf.reduce_any(tf.math.equal(self.labels_l, tf.constant(dim))),
                            dtype=tf.float32),
                            current_log_likelihood)
            else:
                log_likelihood, trans_params = tfa.text.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
                self.trans_params = trans_params  # need to evaluate it for decoding
            # mask = tf.not_equal(self.labels, self.config.vocab_tags.token2id[O])
            # log_likelihood = tf.boolean_mask(log_likelihood, mask)
            if self.config.with_l_id and not self.config.multilang:
                mask = tf.sequence_mask(self.sequence_lengths)
                class_weight = tf.constant([self.config.class_weight, 1 - self.config.class_weight])
                weight_per_label = tf.reshape(tf.gather(class_weight, tf.reshape(self.labels_l, shape=[-1])),
                                              shape=[-1, self.nsteps])
                weight_per_label = tf.reduce_mean(tf.multiply(tf.cast(mask, dtype=tf.float32), weight_per_label),
                                                  axis=1)
                self.loss = tf.reduce_mean(-tf.multiply(log_likelihood, weight_per_label))
            else:
                self.loss = tf.reduce_mean(-log_likelihood)

            if self.config.with_l_id:
                with tf.compat.v1.variable_scope("l_id_loss"):
                    log_likelihood_l, trans_params_l = tfa.text.crf.crf_log_likelihood(
                        self.logits_l, self.labels_l, self.sequence_lengths)
                    self.trans_params_l = trans_params_l  # need to evaluate it for decoding
                    if not self.config.multilang:
                        self.loss += self.config.l_id_weight * tf.reduce_mean(-tf.multiply(log_likelihood_l,
                                                                                           weight_per_label))
                    else:
                        self.loss += tf.reduce_mean(-log_likelihood_l)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            # l_mask = tf.not_equal(self.labels, self.config.vocab_tags.token2id[O])
            if self.config.with_l_id and not self.config.multilang:
                class_weight = tf.constant([self.config.class_weight, 1 - self.config.class_weight])
                weight_per_label = tf.reshape(tf.gather(class_weight, tf.reshape(self.labels_l, shape=[-1])),
                                              shape=[-1, self.nsteps])
                losses = tf.multiply(losses, weight_per_label)
            losses = tf.boolean_mask(losses, mask)
            # losses = tf.boolean_mask(losses, l_mask)
            self.loss = tf.reduce_mean(losses)
            if self.config.with_l_id:
                losses_l = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_l, labels=self.labels_l)
                if not self.config.multilang:
                    losses_l = tf.multiply(losses_l, weight_per_label)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses_l = tf.boolean_mask(losses_l, mask)
                self.loss += self.config.l_id_weight * tf.reduce_mean(losses_l)

        # Add L2 regularisation
        # tv = tf.trainable_variables()
        # regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if v.name not in ['b', 'b1', 'b2', 'bout']])
        # self.loss += 0.05 * regularization_cost

        # for tensorboard
        tf.compat.v1.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, words):
        """
        :param words: list of sentences
        :return labels_pred: list of labels for each sentence sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = [] if not self.config.multilang else [list() for _ in range(self.config.nlangs)]
            if self.config.with_l_id:
                viterbi_l_ids = []
                logits, logits_l, trans_params, trans_params_l, nsteps = self.sess.run(
                    [self.logits, self.logits_l, self.trans_params, self.trans_params_l, self.nsteps],
                    feed_dict=fd)
                for logit, sequence_length in zip(logits_l, sequence_lengths):
                    logit = logit[:sequence_length]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tfa.text.crf.viterbi_decode(
                        logit, trans_params_l)
                    viterbi_l_ids += [viterbi_seq]
            else:
                logits, trans_params, nsteps = self.sess.run(
                    [self.logits, self.trans_params, self.nsteps], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            if self.config.multilang:
                for dim in range(self.config.nlangs):
                    current_logits = np.reshape(logits[:, :, dim:dim + 1, :],
                                                newshape=[-1, nsteps, self.config.ntags])
                    for logit, sequence_length in zip(current_logits, sequence_lengths):
                        logit = logit[:sequence_length]  # keep only the valid steps
                        viterbi_seq, viterbi_score = tfa.text.crf.viterbi_decode(
                            logit, trans_params[dim])
                        viterbi_sequences[dim] += [viterbi_seq]
            else:
                for logit, sequence_length in zip(logits, sequence_lengths):
                    logit = logit[:sequence_length]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tfa.text.crf.viterbi_decode(
                        logit, trans_params)
                    viterbi_sequences += [viterbi_seq]

            if self.config.with_l_id:
                if self.config.multilang:
                    viterbi_sequences = [np.asarray(list(inner)).T for inner in np.asarray(viterbi_sequences).T]
                return viterbi_sequences, viterbi_l_ids, sequence_lengths
            else:
                return viterbi_sequences, sequence_lengths

        else:
            if self.config.with_l_id:
                labels_pred, labels_pred_l = self.sess.run([self.labels_pred, self.labels_pred_l], feed_dict=fd)

                return labels_pred, labels_pred_l, sequence_lengths
            else:
                labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

                return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        :param train: dataset that yields tuple of sentences, tags
        :param dev: dataset
        :param epoch: (int) index of the current epoch
        :return acc: (python float), score to select model on, higher is better
        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        if self.config.shuffle:
            np.random.seed(self.config.seed)
            train = np.random.permutation(np.asarray(list(train)))
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels, l_id) in enumerate(minibatches(train, batch_size)):
            if not self.config.with_l_id:
                l_id = None
            fd, _ = self.get_feed_dict(words, labels=labels, labels_l=l_id, lr=self.config.lr,
                                       dropout=self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["acc"]

    def run_evaluate(self, test):
        """Evaluates performance on test set

        :param test: dataset that yields tuple of (sentences, tags)
        :return metrics: (dict) metrics["acc"] = 74.6, ...
        """
        accs = [] if not self.config.multilang else [[] for _ in range(self.config.nlangs)]
        if self.config.multilang:
            lids = []
        saccs = []
        accs_l = []
        saccs_l = []
        for words, labels, l_id in minibatches(test, self.config.batch_size):
            if self.config.with_l_id:
                labels_pred, labels_pred_l, sequence_lengths = self.predict_batch(words)
            else:
                labels_pred, sequence_lengths = self.predict_batch(words)
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab_pred = np.asarray(lab_pred)
                lab = lab[:length]
                if self.config.multilang and lab_pred.shape[0] == self.config.nlangs and lab_pred.shape[1] == length:
                    lab_pred = lab_pred.T
                lab_pred = lab_pred[:length]
                if self.config.multilang:
                    for a, b in zip(lab, lab_pred):
                        for i in range(self.config.nlangs):
                            accs[i].append(a[i] == b[i])
                    saccs.append(all([t1 == t2 for (a, b) in zip(lab, lab_pred) for (t1, t2) in zip(a, b)]))
                else:
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]
                    saccs.append(all([a == b for (a, b) in zip(lab, lab_pred)]))
            if self.config.with_l_id:
                for lab, lab_pred, length in zip(l_id, labels_pred_l, sequence_lengths):
                    if self.config.multilang:
                        lids += lab
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs_l += [a == b for (a, b) in zip(lab, lab_pred)]
                    saccs_l.append(all([a == b for (a, b) in zip(lab, lab_pred)]))

        if self.config.multilang:
            normed_accs = [acc[lid] for acc, lid in zip(np.asarray(accs).T, lids)]
            normed_pl_accs = [acc[lid] for acc, lid in zip(np.asarray(accs).T, lids)
                              if lid not in self.config.non_pl_lang_ids]
        acc = np.mean(accs) if not self.config.multilang else \
            np.asarray([np.mean([acc[dim] for acc, lid in zip(np.asarray(accs).T, lids) if lid == dim])
                        for dim in range(self.config.nlangs)])
        sacc = np.mean(saccs)
        if self.config.with_l_id:
            if self.config.multilang:
                normed_acc = np.mean(normed_accs)
                normed_pl_acc = np.mean(normed_pl_accs)
            acc_l = np.mean(accs_l)
            join_acc = np.mean(accs + accs_l) if not self.config.multilang else np.mean(normed_accs + accs_l)
            sacc_l = np.mean(saccs_l)
            join_sacc = np.mean(saccs + saccs_l)

            if self.config.multilang:
                metrics = {
                    "acc": 100 * normed_acc,
                    "acc_pl": 100 * normed_pl_acc,
                    "acc_l_id": 100 * acc_l,
                    "joint_acc": 100 * join_acc,
                    "sent_acc": 100 * sacc,
                    "sent_acc_l_id": 100 * sacc_l,
                    "sent_joint_acc": 100 * join_sacc, }
                for i in range(self.config.nlangs):
                    metrics['acc_%d' % i] = 100 * acc[i]
                return metrics
            else:
                return {"acc": 100 * acc,
                        "acc_l_id": 100 * acc_l,
                        "joint_acc": 100 * join_acc,
                        "sent_acc": 100 * sacc,
                        "sent_acc_l_id": 100 * sacc_l,
                        "sent_joint_acc": 100 * join_sacc,
                        }
        else:
            return {"acc": 100 * acc, "sent_acc": 100 * sacc}

    def predict(self, words_raw):
        """Returns list of tags

        :param words_raw: list of words (string), just one sentence (no batch)
        :return preds: list of tags (string), one for each word in the sentence
        """
        words = [self.config.processing_word(w) for w in words_raw]
        if len(words) > 0:
            if type(words[0]) == tuple:
                words = list(zip(*words))
            pred = self.predict_batch([words])
            pred_ids = np.asarray(pred[0])
            if self.config.multilang:
                preds = [[self.idx_to_tag[idx] for idx in list(pred_ids.T[0][n])] for n in range(self.config.nlangs)]
                preds = [list(inner) for inner in np.asarray(preds).T]
            else:
                preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

            if self.config.with_l_id:
                pred_lid = pred[1][0]
                if self.config.multilang:
                    resulting_lids = list()
                    for lid in pred_lid:
                        resulting_lids.append(self.config.id_to_lang[lid])
                    pred_lid = resulting_lids
                return preds, pred_lid
            else:
                return preds
        else:
            if self.config.with_l_id:
                return [], []
            else:
                return []
