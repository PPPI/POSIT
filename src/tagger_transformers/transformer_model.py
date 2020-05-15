import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nltk import casual_tokenize
from tensorflow.python.keras.models import save_model, load_model

from ..preprocessor.preprocess import CODE_TOKENISATION_REGEX
from ..tagger.base_model import BaseModel
from ..tagger.general_utils import Progbar


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, -1), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, 4*dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn_weights_block = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights_block


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, 4*dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block = self.enc_layers[i](x, training, mask)
            attention_weights['encoder_layer{}_block'.format(i + 1)] = block
        return x, attention_weights  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, positions_to_enc, rate=0.1,):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(positions_to_enc, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, target_Id_size, positions_to_encode, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, positions_to_encode, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.final_layer_Id = tf.keras.layers.Dense(target_Id_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output, _ = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output_Id = self.final_layer_Id(dec_output)  # (batch_size, tar_seq_len, target_Id_size)

        return final_output, final_output_Id, attention_weights


class TransformerEncOnly(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, target_Id_size, positions_to_encode, rate=0.1):
        super(TransformerEncOnly, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.final_layer_Id = tf.keras.layers.Dense(target_Id_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output, enc_attention_weights = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        final_output_Id = self.final_layer_Id(enc_output)  # (batch_size, tar_seq_len, target_Id_size)

        return final_output, final_output_Id, enc_attention_weights


class TransformerCoSModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_vocab_size = config.nwords + 2
        self.target_vocab_size = config.ntags + 2
        self.transformer = TransformerEncOnly(self.config.num_layers, self.config.d_model, self.config.num_heads,
                                       self.config.dff, self.input_vocab_size, self.target_vocab_size,
                                       5, # Code or English + 2 special start and end tokens repc. + offset of 1 (hardcoded for now)
                                       self.config.max_len, self.config.dropout_rate)
        self.optimizer = tf.keras.optimizers.Adam(config.lr, beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-6)
        # self.optimizer = tf.keras.optimizers.RMSprop(config.lr)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.validate_loss = tf.keras.metrics.Mean(name='test_loss')
        self.validate_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.train_loss_id = tf.keras.metrics.Mean(name='train_loss_id')
        self.train_accuracy_id = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_id')
        self.validate_loss_id = tf.keras.metrics.Mean(name='test_loss_id')
        self.validate_accuracy_id = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_id')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_path = os.path.join('.', self.config.dir_output, "checkpoints", "train")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.token2id.items()}
        self.saver = save_model
        self.loader = load_model
        self.input_start_token_id = self.input_vocab_size - 2
        self.input_end_token_id = self.input_vocab_size - 1
        self.output_start_token_id = self.target_vocab_size - 2
        self.output_end_token_id = self.target_vocab_size - 1
        self.train_nbatches = None
        self.test_nbatches = None
        self.writer = tf.summary.create_file_writer(config.dir_output + "tensorboard")

    def add_start_and_end_python(self, inpt, tar, tarId):
        inpt = np.insert(np.insert(inpt.numpy(), 0, self.input_start_token_id, axis=0), -1, self.input_end_token_id, axis=0).astype(dtype=np.float32)
        tar = np.insert(np.insert(tar.numpy(), 0, self.output_start_token_id, axis=0), -1, self.output_end_token_id, axis=0).astype(dtype=np.float32)
        tarId = np.insert(np.insert(tarId.numpy(), 0, 3, axis=0), -1, 4, axis=0).astype(dtype=np.float32)
        return inpt, tar, tarId
    
    def add_start_and_end(self, inpt, tar, tarId):
        return tf.py_function(self.add_start_and_end_python, [inpt, tar, tarId], [tf.int64, tf.int64, tf.int64])

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, -1))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    @tf.function
    def train_step(self, step, inp, tar, tarId):
        with self.writer.as_default():
            tar_inp = tar[:, :]
            tar_real = tar[:, :]
            tarId_inp = tarId[:, :]
            tarId_real = tarId[:, :]

            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, predictions_Id, _ = self.transformer(inp, tar_inp,
                                                True,
                                                enc_padding_mask,
                                                combined_mask,
                                                dec_padding_mask)
                loss = self.loss_function(tar_real, predictions)
                loss_Id = self.loss_function(tarId_real, predictions_Id)
                full_loss = 0.4*loss + 0.6*loss_Id
            
            gradients = tape.gradient(full_loss, self.transformer.trainable_variables)
            # tf.summary.histogram("gradients", gradients, step=step)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)
            self.train_loss_id(loss_Id)
            self.train_accuracy_id(tarId_real, predictions_Id)
            
            # if (step % 5 == 0):
            #     tf.summary.scalar("training_loss_lid", self.train_loss_id.result(), step=step)
            #     tf.summary.scalar("training_accuracy", self.train_accuracy.result(), step=step)
            #     tf.summary.scalar("training_loss", self.train_loss.result(), step=step)
            #     tf.summary.scalar("training_accuracy_lid", self.train_accuracy_id.result(), step=step)

    @tf.function
    def validate_step(self, step, inp, tar, tarId):
        with self.writer.as_default():
            tar_inp = tar[:, :]
            tar_real = tar[:, :]
            tarId_inp = tarId[:, :]
            tarId_real = tarId[:, :]

            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

            predictions, predictions_Id, _ = self.transformer(inp, tar_inp,
                                            True,
                                            enc_padding_mask,
                                            combined_mask,
                                            dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)
            loss_Id = self.loss_function(tarId_real, predictions_Id)


            self.validate_loss(loss)
            self.validate_accuracy(tar_real, predictions)
            self.validate_loss_id(loss_Id)
            self.validate_accuracy_id(tarId_real, predictions_Id)
            
            # if (step % 5 == 0):
            #     tf.summary.scalar("validation_loss_lid", 
            #     self.validate_loss_id.result(), step=step)
            #     tf.summary.scalar("validation_accuracy", self.validate_accuracy.result(), step=step)
            #     tf.summary.scalar("validation_loss", self.validate_loss.result(), step=step)
            #     tf.summary.scalar("validation_accuracy_lid", self.validate_accuracy_id.result(), step=step)

    def restore_session(self, dir_model):
        """Reload weights into session

        :param dir_model: dir with weights
        """
        self.logger.info("Reloading the latest trained model...")
        self.transformer = self.loader(dir_model)

    def save_session(self):
        """Saves session"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver(self.transformer, self.config.dir_model)

    def train(self, train, dev):
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.logger.info('Latest checkpoint restored!!')
        best_score = 0
        nr_epochs_no_imprvmt = 0  # for early stopping
        pad_shape = ([-1], [-1], [-1]) if self.config.with_l_id else ([-1], [-1])
        # pad_val = (tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64)) if self.config.with_l_id else (tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64))
        train = train\
            .map(self.add_start_and_end)\
            .shuffle(self.config.buffer_size, seed=self.config.seed)\
            .padded_batch(self.config.batch_size,padded_shapes=pad_shape)
        self.train_nbatches = len(list(train)) if self.train_nbatches is None else self.train_nbatches
        prog = Progbar(target=self.train_nbatches)

        for epoch in range(self.config.nr_epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_loss_id.reset_states()
            self.train_accuracy_id.reset_states()

            # inp -> sents, tar -> tags
            for (batch, (inp, tar, tarId)) in enumerate(train):
                self.train_step(self.train_nbatches*epoch+batch, inp, tar, tarId)
                self.writer.flush()
                if (batch == self.train_nbatches) or (batch % 2 == 0):
                    prog.update(batch + 1,
                                exact=[("train loss", self.train_loss.result()), ("train accuracy", self.train_accuracy.result()), ("train loss id", self.train_loss_id.result()), ("train accuracy id", self.train_accuracy_id.result())])
            prog.update(self.train_nbatches,
                        exact=[("train loss", self.train_loss.result()), ("train accuracy", self.train_accuracy.result()), ("train loss id", self.train_loss_id.result()), ("train accuracy id", self.train_accuracy_id.result())])
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                self.logger.info('Saving checkpoint for epoch {} at {}'
                .format(epoch + 1, ckpt_save_path))

            self.logger.info(
                '\nEpoch {} Loss {:.4f} Accuracy {:.4f}'
                    .format(epoch + 1, self.train_loss.result(), self.train_accuracy.result())
            )

            self.logger.debug('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            if dev is not None:
                self.validate(dev)
                score = self.validate_accuracy.result()
                # early stopping and saving best parameters
                if score >= best_score:
                    nr_epochs_no_imprvmt = 0
                else:
                    nr_epochs_no_imprvmt += 1
                    if nr_epochs_no_imprvmt >= self.config.nr_epochs_no_imprvmt:
                        self.logger.info("- early stopping {} epochs without "
                                        "improvement".format(nr_epochs_no_imprvmt))
                        ckpt_save_path = self.ckpt_manager.save()
                        self.logger.info('Saving checkpoint for epoch {} at {}'
                        .format(epoch + 1, ckpt_save_path))
                        break

    def validate(self, test):
        # if a checkpoint exists, restore the latest checkpoint.
        start = time.time()

        self.validate_loss.reset_states()
        self.validate_accuracy.reset_states()
        self.validate_loss_id.reset_states()
        self.validate_accuracy_id.reset_states()

        pad_shape = ([-1], [-1], [-1]) if self.config.with_l_id else ([-1], [-1])
        # pad_val = (tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64)) if self.config.with_l_id else (tf.constant(-1, dtype=tf.int64), tf.constant(-1, dtype=tf.int64))
        test = test\
            .map(self.add_start_and_end)\
            .shuffle(self.config.buffer_size, seed=self.config.seed)\
            .padded_batch(self.config.batch_size, padded_shapes=pad_shape)
        self.test_nbatches = len(list(test)) if self.test_nbatches is None else self.test_nbatches
        prog = Progbar(target=self.test_nbatches)

        # inp -> sents, tar -> tags
        for (batch, (inp, tar, tarId)) in enumerate(test):
            self.validate_step(batch, inp, tar, tarId)
            self.writer.flush()
            prog.update(batch + 1,
                        exact=[("validate loss", self.validate_loss.result()), ("validate accuracy", self.validate_accuracy.result()), ("validate loss id", self.validate_loss_id.result()), ("validate accuracy id", self.validate_accuracy_id.result())])

        self.logger.info(
            '\n[Validation]: Loss {:.4f} Accuracy {:.4f}'
                .format(self.validate_loss.result(), self.validate_accuracy.result())
        )

        self.logger.debug('Time taken for validation: {} secs\n'.format(time.time() - start))

    def evaluate(self, inp_sentence):
        start_token = [self.input_vocab_size - 1]
        end_token = [self.input_vocab_size - 2]

        # inp sentence is english
        inp_sentence = start_token + [self.config.processing_word(w) for w in inp_sentence] + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is tags, the tags start token is the start of the decode state
        decoder_input = [self.target_vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.config.max_len):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, predictions_id, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.target_vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] + [i for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.idx_to_tag[i] for i in result
                                if i < self.target_vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()

    def tag(self, sentence, plot='', casual=False):
        if casual:
            sentence = casual_tokenize(sentence.strip())
        else:
            sentence = [l.strip()
                        for l in re.findall(CODE_TOKENISATION_REGEX,
                                            sentence.strip())
                        if len(l.strip()) > 0]
        result, result_id, attention_weights = self.evaluate(sentence)

        predicted_sentence = ' '.join([self.idx_to_tag[i] for i in result if i < self.target_vocab_size])

        print('Input: {}'.format(' '.join(sentence)))
        print('Predicted tags: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)
