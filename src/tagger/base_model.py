import os

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


class BaseModel(object):
    """Generic class for general methods that are not specific to PoS tagger."""

    def __init__(self, config):
        """Defines self.config and self.logger

        :param config: (Config instance) class with hyper parameters, vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.use_cpu = config.use_cpu
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitialises the weights of a given layer
        :param scope_name: The name of the scope within which to reinitilise"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, optimiser, lr, loss, clip=None):
        """Defines self.train_op that performs an update on a batch

        :param optimiser: (string) sgd method, for example "adam"
        :param lr: (tf.placeholder) tf.float32, learning rate
        :param loss: (tensor) tf.float32 loss to minimize
        :param clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _optimiser_lower = optimiser.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _optimiser_lower == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _optimiser_lower == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _optimiser_lower == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _optimiser_lower == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            elif _optimiser_lower == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(lr)
            elif _optimiser_lower == 'proximaladagrad':
                optimizer = tf.train.ProximalAdagradOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_optimiser_lower))

            if clip is not None:  # gradient clipping if clip is defined
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        if self.use_cpu:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            # config.log_device_placement = True  # to log device placement (on which device the operation ran)
            #                                     # (nothing gets printed in Jupyter, only if you run it standalone)

        # XXX: Non-adam optimisers have a TEMP VAR reuse issue on TF 1.13+
        # Disabling arithmetic operation optimisations makes the model run
        # It has been accessed by other users (on different models) that
        # this issue can manifest as model sizes are increased.
        # On 1.14+, memory optimisation also causes issues.
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off
        config.graph_options.rewrite_options.memory_optimization = off

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session

        :param dir_model: dir with weights
        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        """Saves session"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                                                 self.sess.graph)

    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        :param train: dataset that yields tuple of (sentences, tags)
        :param dev: dataset
        """
        best_score = 0
        nr_epochs_no_imprvmt = 0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.config.nr_epochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                                                           self.config.nr_epochs))

            score = self.run_epoch(train, dev, epoch)
            if isinstance(self.config.lr, float):
                self.config.lr *= self.config.lr_decay  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nr_epochs_no_imprvmt = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nr_epochs_no_imprvmt += 1
                if nr_epochs_no_imprvmt >= self.config.nr_epochs_no_imprvmt:
                    self.logger.info("- early stopping {} epochs without "
                                     "improvement".format(nr_epochs_no_imprvmt))
                    break

    def evaluate(self, test):
        """Evaluate model on test set

        :param test: dataset that yields tuple of (sentences, tags)
        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)
