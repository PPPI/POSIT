import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
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

        with tf.compat.v1.variable_scope("train_step"):
            if _optimiser_lower == 'adam':  # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _optimiser_lower == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _optimiser_lower == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
            elif _optimiser_lower == 'rmsprop':
                optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
            elif _optimiser_lower == 'adadelta':
                optimizer = tf.compat.v1.train.AdadeltaOptimizer(lr)
            elif _optimiser_lower == 'proximaladagrad':
                optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_optimiser_lower))

            if clip is not None:  # gradient clipping if clip is defined
                grads, vs = zip(
                    *optimizer.compute_gradients(loss,
                                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss,
                                                   aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf.compat.v1.Session")
        if self.use_cpu:
            config = tf.compat.v1.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            # config.log_device_placement = True  # to log device placement (on which device the operation ran)
            #                                     # (nothing gets printed in Jupyter, only if you run it standalone)

        tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
        if not self.config.use_cpu:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        # Make use of XLA
        tf.config.optimizer.set_jit(True)

        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

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
        self.merged = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self.config.dir_output,
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
