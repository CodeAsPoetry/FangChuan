# coding="utf-8"

import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from random import randint

batchSize=512
numClasses=2
maxSeqLength=100
wordDim = 300
hiddenSize = 150
DELAT=0.2

corpusWord2Vect="C:/PyWorkSpace/FangChuan/myWordVec/w2v_chisim.300d.txt"

Parent_File_Path="C:/ModelWorkPath/corpus_60000/"

Pos_Txt_Index_List_Path=Parent_File_Path+"Pos_Txt_Index.npy"
Neg_Txt_Index_List_Path=Parent_File_Path+"Neg_Txt_Index.npy"

allVocabListNPYPath = "C:/PyWorkSpace/FangChuan/myWordVec/allVocabList.npy"
allVocabList = list(np.load(allVocabListNPYPath))

Total_Sample_Num = 120000

model_path = 'C://ModelWorkPath/Model_ZH/'
log_dir = 'C://ModelWorkPath/PaperPic'

def getSplitSets():
    Train_Set=[]
    Valid_Set=[]
    Test_Set=[]
    for i in range(10000):
        for j in range(12):
            if j < 10:
                Train_Set.append(12 * i + j)
            else:
                if j == 10:
                    Valid_Set.append(12 * i + j)
                if j == 11:
                    Test_Set.append(12 * i + j)

    return Train_Set,Valid_Set,Test_Set

def getTrainBatch(Train_Set,Pos_Txt_Index_List,Neg_Txt_Index_List):

    TrainBatchSampleIndex=[]

    TrainBatchWordIndex=[]

    TrainBatchWordVec=[]

    TrainBatchLabel=[]

    for i in range(batchSize):
        num = randint(0, 120000)
        while(num not in Train_Set):
            num = randint(0, 120000)
        TrainBatchSampleIndex.append(num)


    for i in range(batchSize):
        the_sample_index=[]
        if TrainBatchSampleIndex[i]<60000:
            the_sample_index_pri = Pos_Txt_Index_List[i]
            TrainBatchLabel.append([1,0])
        else:
            the_sample_index_pri = Neg_Txt_Index_List[i-60000]
            TrainBatchLabel.append([0,1])

        if len(the_sample_index_pri)>=maxSeqLength:
            for i in range(maxSeqLength):
                the_sample_index.append(the_sample_index_pri[i])
        else:
            while(len(the_sample_index_pri)<maxSeqLength):
                the_sample_index_pri.append(0)
            the_sample_index = the_sample_index_pri

        TrainBatchWordIndex.append(the_sample_index)

    for i in range(batchSize):
        the_sample_vec=[]
        the_sample_index = TrainBatchWordIndex[i]
        for j in range(maxSeqLength):
            the_sample_vec.append(model.wv[allVocabList[the_sample_index[j]]])
        TrainBatchWordVec.append(the_sample_vec)

    TrainBatchWordVec = np.array(TrainBatchWordVec)

    TrainBatchLabel = np.array(TrainBatchLabel)

    return TrainBatchSampleIndex,TrainBatchWordVec,TrainBatchLabel

def getValidBatch(Valid_Set,Pos_Txt_Index_List,Neg_Txt_Index_List):
    ValidBatchSampleIndex = []

    ValidBatchWordIndex = []

    ValidBatchWordVec = []

    ValidBatchLabel = []

    for i in range(batchSize):
        num = randint(0, 120000)
        while (num not in Valid_Set):
            num = randint(0, 120000)
        ValidBatchSampleIndex.append(num)

    for i in range(batchSize):
        the_sample_index = []
        if ValidBatchSampleIndex[i] < 60000:
            the_sample_index_pri = Pos_Txt_Index_List[i]
            ValidBatchLabel.append([1, 0])
        else:
            the_sample_index_pri = Neg_Txt_Index_List[i - 60000]
            ValidBatchLabel.append([0, 1])

        if len(the_sample_index_pri) >= maxSeqLength:
            for i in range(maxSeqLength):
                the_sample_index.append(the_sample_index_pri[i])
        else:
            while (len(the_sample_index_pri) < maxSeqLength):
                the_sample_index_pri.append(0)
            the_sample_index = the_sample_index_pri

        ValidBatchWordIndex.append(the_sample_index)

    for i in range(batchSize):
        the_sample_vec = []
        the_sample_index = ValidBatchWordIndex[i]
        for j in range(maxSeqLength):
            the_sample_vec.append(model.wv[allVocabList[the_sample_index[j]]])
        ValidBatchWordVec.append(the_sample_vec)

    ValidBatchWordVec = np.array(ValidBatchWordVec)

    ValidBatchLabel = np.array(ValidBatchLabel)

    return ValidBatchSampleIndex,ValidBatchWordVec, ValidBatchLabel

def getTestBatch(Test_Set,Pos_Txt_Index_List,Neg_Txt_Index_List):
    TestBatchSampleIndex = []

    TestBatchWordIndex = []

    TestBatchWordVec = []

    TestBatchLabel = []

    for i in range(batchSize):
        num = randint(0, 120000)
        while (num not in Test_Set):
            num = randint(0, 120000)
        TestBatchSampleIndex.append(num)

    for i in range(batchSize):
        the_sample_index = []
        if TestBatchSampleIndex[i] < 60000:
            the_sample_index_pri = Pos_Txt_Index_List[i]
            TestBatchLabel.append([1.0, 0.0])
        else:
            the_sample_index_pri = Neg_Txt_Index_List[i - 60000]
            TestBatchLabel.append([0.0, 1.0])

        if len(the_sample_index_pri) >= maxSeqLength:
            for i in range(maxSeqLength):
                the_sample_index.append(the_sample_index_pri[i])
        else:
            while (len(the_sample_index_pri) < maxSeqLength):
                the_sample_index_pri.append(0)
            the_sample_index = the_sample_index_pri

        TestBatchWordIndex.append(the_sample_index)

    for i in range(batchSize):
        the_sample_vec = []
        the_sample_index = TestBatchWordIndex[i]
        for j in range(maxSeqLength):
            the_sample_vec.append(model.wv[allVocabList[the_sample_index[j]]])
        TestBatchWordVec.append(the_sample_vec)

    TestBatchWordVec = np.array(TestBatchWordVec)

    TestBatchLabel = np.array(TestBatchLabel)

    return TestBatchSampleIndex,TestBatchWordVec, TestBatchLabel

if __name__=="__main__":
    print("CASDMN_Model")

    model = Word2Vec.load(corpusWord2Vect)
    Pos_Txt_Index_List = list(np.load(Pos_Txt_Index_List_Path))
    Neg_Txt_Index_List = list(np.load(Neg_Txt_Index_List_Path))


    Train_Set, Valid_Set,Test_Set = getSplitSets()


    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_text = tf.placeholder(tf.float32, [batchSize, maxSeqLength,wordDim])
    input_emoji = tf.placeholder(tf.float32,[batchSize,wordDim])

    # (Bi-)RNN layer(-s)
    seq_len_ph = []
    for i in range(batchSize):
        seq_len_ph.append(maxSeqLength)
    rnn_outputs, _ = bi_rnn(GRUCell(hiddenSize), GRUCell(hiddenSize),
                            inputs=input_text, sequence_length=seq_len_ph, dtype=tf.float32)

    memory = tf.concat(rnn_outputs, 2)

    attention_input_1 = tf.reduce_mean(input_text, axis=1)



    def attention(memory, input):

        input = tf.reshape(input, [batchSize, 1, wordDim])

        inputs = input
        for i in range(memory.shape[1] - 1):
            inputs = tf.concat((inputs, input), 1)

        inputss = tf.concat((memory, inputs), 2)

        inputss = tf.transpose(inputss, [0, 2, 1])
        w = tf.Variable(tf.random_uniform([wordDim * 2, 1], -1.0, 1.0))
        b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        alphas = tf.nn.softmax(tf.tanh(tf.tensordot(inputss, w, axes=([1], [0])) + b))

        attention_out = np.dot(memory, alphas)
        attention_out = tf.reduce_mean(attention_out, 1)
        input = tf.reshape(input, [batchSize, wordDim])
        attention_out = (1 - DELAT) * attention_out + DELAT * input
        return attention_out, alphas


    output_1, alphas_1 = attention(memory, attention_input_1)

    output_1 = tf.nn.dropout(output_1, 0.75)



    weight = tf.Variable(tf.truncated_normal([wordDim, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    prediction = tf.nn.softmax(tf.matmul(output_1, weight) + bias)


    # with tf.name_scope('cross_entropy'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    # tf.summary.scalar('cross_entropy',loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    tf.summary.scalar('accuracy',accuracy)



    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir+'/test')
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        print("Traing starting...")

        for i in range(1000):
            _,TrainBatchWordVec, TrainBatchLabel = getTrainBatch(Train_Set, Pos_Txt_Index_List, Neg_Txt_Index_List)

            summary,_ = sess.run([merged,optimizer], {input_text: TrainBatchWordVec, labels: TrainBatchLabel})
            train_writer.add_summary(summary,i)

            acc = sess.run(accuracy, feed_dict={input_text: TrainBatchWordVec, labels: TrainBatchLabel})
            coat = sess.run(loss, feed_dict={input_text: TrainBatchWordVec, labels: TrainBatchLabel})
            if i % 10 == 0:
                print('Iter' + str(i * batchSize) + ", Minibatch Loss=" + \
                      '{:.6f}'.format(coat) + ", Training Accuracy=" + \
                      "{:.5f}".format(acc))
            if i % 10 == 0:
                _,ValidBatchWordVec, ValidBatchLabel = getValidBatch(Valid_Set, Pos_Txt_Index_List, Neg_Txt_Index_List)
                save_path = saver.save(sess, model_path + "text_bi_att.ckpt", global_step=i)
                coat_valid = sess.run(loss, feed_dict={input_text: ValidBatchWordVec, labels: ValidBatchLabel})
                summary,acc_valid = sess.run([merged,accuracy], {input_text: ValidBatchWordVec, labels: ValidBatchLabel})
                print('Iter' + str(i * batchSize) + ", Minibatch Loss=" + \
                      '{:.6f}'.format(coat_valid) + ", Validing Accuracy=" + \
                      "{:.5f}".format(acc_valid))
                test_writer.add_summary(summary,i)

        print("Training finished!")


        print("Testing start")

        for i in range(10):
            _,TestBatchWordVec, TestBatchLabel = getTestBatch(Test_Set, Pos_Txt_Index_List, Neg_Txt_Index_List)
            acc = sess.run(accuracy, feed_dict={input_text: TestBatchWordVec, labels: TestBatchLabel})
            coat = sess.run(loss, feed_dict={input_text: TestBatchWordVec, labels: TestBatchLabel})
            print('Iter' + str(i * batchSize) + ", Minibatch Loss=" + \
                  '{:.6f}'.format(coat) + ", Testing Accuracy=" + \
                  "{:.5f}".format(acc))

        print("Testing finished!")

        train_writer.close()
        test_writer.close()

