# coding = 'utf-8'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from MyDeepLearningModel.attention import attention
from MyDeepLearningModel.utils import get_vocabulary_size

MODEL_PATH = 'myModel/model-300'

# Load the data set
posIndexListPath = 'posIndexList.npy'
negIndexListPath = 'negIndexList.npy'
posIndexList = np.load(posIndexListPath)
posLabel = np.ones([60000, ])
negIndexList = np.load(negIndexListPath)
negLabel = np.zeros([60000, ])
indexList = (posIndexList, negIndexList)
labelList = (posLabel, negLabel)
X = np.concatenate(indexList, 0)
# Sequences pre-processing
vocabulary_size = get_vocabulary_size(X)


g = tf.Graph()
sess = tf.Session(graph=g)
with sess.as_default():
    with g.as_default():
        SEQUENCE_LENGTH = 250
        EMBEDDING_DIM = 100
        HIDDEN_SIZE = 150
        ATTENTION_SIZE = 50
        KEEP_PROB = 0.8
        with tf.name_scope('Inputs'):
            batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
            target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
            seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
            tf.summary.histogram('embeddings_var', embeddings_var)
            batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

        # (Bi-)RNN layer(-s)
        rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                                inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
        tf.summary.histogram('RNN_outputs', rnn_outputs)
        rnn_outputs_cat = tf.concat(rnn_outputs, 2)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        # Dropout
        drop = tf.nn.dropout(attention_output, KEEP_PROB)

        # Fully connected layer
        with tf.name_scope('Fully_connected_layer'):
            W = tf.Variable(
                tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
            b = tf.Variable(tf.constant(0., shape=[1]))
            y_hat = tf.nn.xw_plus_b(drop, W, b)
            y_hat = tf.squeeze(y_hat)
            tf.summary.histogram('W', W)

            pri_y_hat = tf.round(tf.sigmoid(y_hat))

        tf.global_variables_initializer().run()
        saver_restore = tf.train.Saver(tf.global_variables())
        saver_restore.restore(sess, MODEL_PATH)

allVocabList = list(np.load('allVocabList.npy'))
# print('allVocabList', len(allVocabList), allVocabList)
allVocabDict = {}
for i in range(len(allVocabList)):
    allVocabDict[allVocabList[i]] = i

sentence = '周末在家被老婆要求给她拍照。'
import jieba

senList = list(jieba.cut(sentence))

senIndex = []
for i in range(len(senList)):
    if senList[i] in allVocabList:
        # print(senList[i])
        senIndex.append(allVocabDict[senList[i]])
    else:
        continue

seq_len = []
seq_len.append(len(senIndex))

while len(senIndex) < 250 :
    senIndex.append(0)

senIndex = np.array(senIndex).reshape([1, 250])



with sess.as_default():
    with sess.graph.as_default():
        pri_y_hat_infer = sess.run(pri_y_hat, feed_dict={batch_ph: senIndex, seq_len_ph: seq_len,})
        print(pri_y_hat_infer)
