#encoding=utf8
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
import os

import cws.BiLSTM as modelDef
from cws.data import Data
#from tips import max_sub_count

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.app.flags.DEFINE_string('dict_path', 'data/your_dict.pkl', 'dict path')
tf.app.flags.DEFINE_string('train_data', 'data/your_train_data.pkl', 'train data path')
tf.app.flags.DEFINE_string('ckpt_path', 'checkpoint/cws.finetune.ckpt/', 'checkpoint path')
tf.app.flags.DEFINE_integer('embed_size', 256, 'embedding size')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'hidden layer node number')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 20, 'training epoch')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('save_path','checkpoint/cws.ckpt/','new model save path')

FLAGS = tf.app.flags.FLAGS

class BiLSTMTrain(object):
    def __init__(self, data_train=None, data_valid=None, data_test=None, model=None):
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.model = model

    def train(self):
       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
       ## finetune ##
       # ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
       # saver = tf.train.Saver()
       # saver.restore(sess, ckpt)
       # print('-->finetune the ckeckpoint:'+ckpt+'...')
       ##############
        max_epoch = 10
        tr_batch_size = FLAGS.batch_size
        max_max_epoch = FLAGS.epoch  # Max epoch
        display_num = 10  # Display 10 pre epoch
        tr_batch_num = int(self.data_train.y.shape[0] / tr_batch_size)  
        display_batch = int(tr_batch_num / display_num)  
        saver = tf.train.Saver(max_to_keep=10)  
        for epoch in range(max_max_epoch): 
            _lr = FLAGS.lr
            if epoch > max_epoch:
                _lr = 0.0002
            print('EPOCH %d， lr=%g' % (epoch + 1, _lr))
            start_time = time.time()
            _losstotal = 0.0
            show_loss = 0.0
            for batch in range(tr_batch_num):  
                fetches = [self.model.loss, self.model.train_op]
                X_batch, y_batch = self.data_train.next_batch(tr_batch_size)

                feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: _lr,
                             self.model.batch_size: tr_batch_size,
                             self.model.keep_prob: 0.5}
                _loss, _ = sess.run(fetches, feed_dict)  
                _losstotal += _loss
                show_loss += _loss
                if (batch + 1) % display_batch == 0:
                    P, R, F1 = self.test_epoch(self.data_valid, sess)  # valid
                    print('\ttraining loss=%g ;  Precision= %g ; Recall= %g ; F1= %g ;' % (show_loss / display_batch,
                                                                             P, R, F1))
                    show_loss = 0.0
            mean_loss = _losstotal / (tr_batch_num + 0.000001)
            if (epoch + 1) % 1 == 0:  # Save once per epoch
                save_path = saver.save(sess, self.model.model_save_path+'_plus', global_step=(epoch + 1))
                print('the save path is ', save_path)
            print('\ttraining %d, loss=%g ' % (self.data_train.y.shape[0], mean_loss))
            print('Epoch training %d, loss=%g, speed=%g s/epoch' % (
                self.data_train.y.shape[0], mean_loss, time.time() - start_time))

        # testing
        print('**TEST RESULT:')
        P, R, F1 = self.test_epoch(self.data_test, sess)
        print('**Test %d, Precision= %g ; Recall= %g ; F1= %g ' % (self.data_test.y.shape[0], P, R, F1))
        sess.close()

    def test_epoch(self, dataset=None, sess=None):
        
        _batch_size = FLAGS.batch_size
        _y = dataset.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)  
        #correct_labels = 0
        #total_labels = 0
        total_correct_count = 0
        total_P_count = 0
        total_R_count = 0
        fetches = [self.model.scores, self.model.length, self.model.transition_params]

        for k in range(batch_num):
            X_batch, y_batch = dataset.next_batch(_batch_size)
            feed_dict = {self.model.X_inputs: X_batch, self.model.y_inputs: y_batch, self.model.lr: 1e-5,
                         self.model.batch_size: _batch_size,
                         self.model.keep_prob: 1.0}

            test_score, test_length, transition_params = sess.run(fetches=fetches,
                                                                  feed_dict=feed_dict)
            #print(test_score)
            #print(test_length)

            for tf_unary_scores_, y_, sequence_length_ in zip(
                    test_score, y_batch, test_length):
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]
                viterbi_sequence, _ = crf.viterbi_decode(
                    tf_unary_scores_, transition_params)
                y_ = y_.tolist()

                sequence_train_count = 0
                sequence_label_count = 0

                for i in range(sequence_length_):
                    if viterbi_sequence[i] == 4:
                        sequence_train_count += 1
                    if viterbi_sequence[i] == 1:
                        sequence_train_count += 1

                for i in range(sequence_length_):
                    if y_[i] == 4:
                        sequence_label_count += 1
                    if y_[i] == 1:
                        sequence_label_count += 1
                viterbi_sequence_str = [str(i) for i in viterbi_sequence]
                viterbi_sequence_str = ''.join(viterbi_sequence_str)
                y_str = [str(i) for i in y_]
                y_str = ''.join(y_str)

                i = 0
                correct_count = 0
                while (i < sequence_length_ ):

                    if y_str[i] == '4':
                        if viterbi_sequence_str[i] == '4':
                            correct_count += 1
                            i += 1
                        else:
                            i += 1
                    else:
                        if y_str[i] == '1':
                            start = i
                            i += 1
                            while(y_str[i] == '2'):
                                i += 1
                            end = i
                            i += 1
                            if y_str[start: end + 1] == viterbi_sequence_str[start: end + 1]:
                                correct_count += 1

                total_correct_count += correct_count
                total_P_count += sequence_train_count
                total_R_count += sequence_label_count


                #correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                #total_labels += sequence_length_

        P = total_correct_count / float(total_P_count)
        R = total_correct_count / float(total_R_count)
        F1 = 2 * P * R / (P + R)
        return P, R, F1

def main(_):
    Data_ = Data(dict_path=FLAGS.dict_path, train_data=FLAGS.train_data)
    print('Corpus loading completed:', FLAGS.train_data)
    data_train, data_valid, data_test = Data_.builderTrainData() 
    print('The training set, verification set, and test set split are completed!')
    model = modelDef.BiLSTMModel(max_len=Data_.max_len,
                                 vocab_size=Data_.word2id.__len__()+1, 
                                 class_num= Data_.tag2id.__len__(), 
                                 model_save_path=FLAGS.save_path, 
                                 embed_size=FLAGS.embed_size,  
                                 hs=FLAGS.hidden_size)
    print('Model definition completed!')
    train = BiLSTMTrain(data_train, data_valid, data_test, model)
    train.train()
    print('Model training completed!')

if __name__ == '__main__':
    tf.app.run()
