# -*- coding: utf-8 -*-
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.reset_default_graph()
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

batch_size = 1

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))


stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()


allVocabList = list(np.load('allVocabList.npy'))
# print('allVocabList', len(allVocabList), allVocabList)
allVocabDict = {}
for i in range(len(allVocabList)):
    allVocabDict[allVocabList[i]] = i

sentence = '这个书包不很好'
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

request.model_spec.name = '190420'
request.model_spec.signature_name = 'prediction_signature'
request.inputs['inputs']['batch_ph'].CopyFrom(tf.contrib.util.make_tensor_proto(senIndex, shape=[batch_size, 250], dtype=tf.float32))
request.inputs['inputs']['seq_len_ph'].CopyFrom(tf.contrib.util.make_tensor_proto(seq_len, shape=[batch_size], dtype=tf.int32))
#request.outputs['inputy'].CopyFrom(tf.contrib.util.make_tensor_proto(y, shape=[200, 1], dtype=tf.float32))
#.SerializeToString()
result = stub.Predict(request, 10.0) # 10 secs timeout
# result1 = stub.Regress(request, 10.0)
print (result)