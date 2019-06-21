# encoding=utf8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


class BiLSTMModel(object):
    def __init__(self, max_len=200, vocab_size=None, class_num=None, model_save_path=None, embed_size=256, hs=512):
        self.timestep_size = self.max_len = max_len
        self.vocab_size = vocab_size
        self.input_size = self.embedding_size = embed_size
        self.class_num = class_num
        self.hidden_size = hs
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.model_save_path = model_save_path

        self.train()

    def train(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')

        with tf.variable_scope('Embeddings'):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.X_inputs)
            self.length = tf.cast(tf.reduce_sum(tf.sign(self.X_inputs), 1), tf.int32)

        with tf.variable_scope("BiLSTM"):
            softmax_w = tf.Variable(tf.truncated_normal([self.hidden_size * 2, self.class_num], stddev=0.1))
            softmax_b = tf.Variable(tf.constant(0.1, shape=[self.class_num]))
            cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
            self.lstm_cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell, self.lstm_cell, self.inputs,
                                                                        sequence_length=self.length, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            bilstm_output = tf.reshape(output, [-1, self.hidden_size * 2])

            #print('The shape of BiLstm Layer output:', bilstm_output.shape)

        with tf.variable_scope('outputs'):
            self.y_pred = tf.matmul(bilstm_output,
                                    softmax_w) + softmax_b  # there is no softmax, reduce the amount of calculation.

            self.scores = tf.reshape(self.y_pred, [-1, self.timestep_size,
                                                   self.class_num])  # [batchsize, timesteps, num_class]
            #print('The shape of Output Layer:', self.scores.shape)
            log_likelihood, self.transition_params = crf.crf_log_likelihood(self.scores, self.y_inputs, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)