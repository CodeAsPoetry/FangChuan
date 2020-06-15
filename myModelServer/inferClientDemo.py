# -*- coding: utf-8 -*-
from grpc.beta import implementations
# import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
# import random
# import linecache
# import skimage.io as io
# import heapq

#
# # 数据类型
# dataType = 'train2017'
# imageParentFilePath = "/home/codeaspoetry/WorkDesk/coco/images/{}/".format(dataType)
# # 数据集样本数
# theSampleNum = 118287
# # 得到图像和对应字幕
# # 用于剔除非3通道图像
# no3ChannelImagesNameFilePath = 'resources/no3ChannelImageName_{}.npy'.format(dataType)
# no3ChannelImagesName = np.load(no3ChannelImagesNameFilePath)
#
# # 获取名词短语
# NPList50UKNPPath = 'resources/NPList50UKNP.npy'
# NPList50UKNP = list(np.load(NPList50UKNPPath))
#
# UKNPIndex = NPList50UKNP.index('UKNP')
# # print(UKNPIndex)          # 3366
#
#
#
# # 获取数据txt文本
# ImageNameCaptionsListFilePath = 'resources/ImageNameCaptionsList_{}.txt'.format(dataType)
# ImageNameCaptionsListFile = open(ImageNameCaptionsListFilePath, 'r')
#
# # 将图片长宽填充到[640, 640], 实验验证CoCo数据集中图片长宽最大值均为640
# def pad_image(image, new_size, mode = 'channel_mean'):
#     h_need_pad = int(new_size[0] - image.shape[0])
#     h_top = int(h_need_pad/2)
#     h_bottom = h_need_pad - h_top
#     w_need_pad = int(new_size[1] - image.shape[1])
#     w_left = int(w_need_pad/2)
#     w_right = w_need_pad - w_left
#     pd_image = np.zeros(shape=new_size, dtype=np.uint8)
#     for i in range(image.shape[-1]):
#         ch_mean = np.mean(image[:, :, i], axis=(0, 1))
#         if mode == 'channel_mean':
#             pd_image[:, :, i] = np.pad(image[:, :, i],
#                                        ((h_top, h_bottom), (w_left, w_right)),
#                                        mode='constant',
#                                        constant_values=ch_mean)
#         elif mode == 'edge':
#             pd_image[:, :, i] = np.pad(image[:, :, i],
#                                        ((h_top, h_bottom), (w_left, w_right)),
#                                        mode='edge')
#         elif mode == 'black':
#             pd_image[:, :, i] = np.pad(image[:, :, i],
#                                        ((h_top, h_bottom), (w_left, w_right)),
#                                        mode='constant',
#                                        constant_values=0)
#         elif mode == 'white':
#             pd_image[:, :, i] = np.pad(image[:, :, i],
#                                        ((h_top, h_bottom), (w_left, w_right)),
#                                        mode='constant',
#                                        constant_values=255)
#     return pd_image
#
# # 定义得到batch数据
# def get_next_batch(batch_size):
#     I_list = []
#     # 随机获取一个batch的图像路径
#     batchIdList = []
#     while(len(batchIdList) < batch_size):
#         # 随机从theSampleNum中取出索引
#         theRow = random.randint(1, theSampleNum)
#         line = linecache.getline(ImageNameCaptionsListFilePath, theRow).replace('\n', '')
#         # 解析出图片的路径，如果包含单通道图片，则不予添加，重新取，直到取够batchSize
#         imgUrl = line.split("@")[0]
#         if imgUrl not in no3ChannelImagesName:
#             batchIdList.append(theRow)
#
#     # 生成images
#     images_list = []
#     for theRow in batchIdList:
#         line = linecache.getline(ImageNameCaptionsListFilePath, theRow).replace('\n', '')
#         i_imgurl = line.split("@")[0]
#         I = io.imread(imageParentFilePath + i_imgurl)
#         I_list.append(I)
#         image_arr = np.array(I)
#         # 将图片填充至统一大小
#         pd_image = pad_image(image_arr, [640, 640, 3], 'black')
#         images_list.append(pd_image)
#
#     # 生成labels
#     labels_list = []
#     for theRow in batchIdList:
#         # 解析得到字幕
#         line = linecache.getline(ImageNameCaptionsListFilePath, theRow).replace('\n', '')
#         captions = line.split("@")[1].split("|")
#         # 初始化label
#         the_label = []
#         for i_index in range(3367):
#             the_label.append(0)
#         # 遍历该样本的每条字幕
#         for caption in captions:
#             # 逐个判断NP集合中的元素是否包含在字幕中，如果包含，则label中相应索引位置为1.0
#             for i_np_index in range(len(NPList50UKNP)):
#                 if NPList50UKNP[i_np_index] in caption:
#                     the_label[i_np_index] = 1.0
#
#         the_label_value = 0
#         for i_value in the_label:
#             the_label_value += i_value
#
#         if(the_label_value < 1.0):
#             the_label[UKNPIndex] = 1.0
#
#         labels_list.append(the_label)
#
#     # 将images和labels转换为numpy数组格式
#     images = np.array(images_list)
#     labels = np.array(labels_list)
#
#     return images, labels, I_list
#
# # 自定义准确率
# def get_batch_acc(predict, labels):
#     # 根据pred得到batch大小和类别数
#     sampleNum = predict.shape[0]
#     dimNum = predict.shape[1]
#     # 统计预测正确的样本个数
#     pre_ok = 0
#     for i in range(sampleNum):
#         # 统计出label中为1的维度数目
#         theSampleTagNum = 0
#         for j in range(dimNum):
#             theSampleTagNum += labels[i][j]
#         # 从predict[i]中找到前theSampleTagNum大的位置索引列表
#         nums = list(predict[i])
#         max_num_index_list = list(map(nums.index, heapq.nlargest(theSampleTagNum, nums)))
#         # 计数变量，遍历得到的前theSampleTagNum大的位置索引列表，如果对应索引上label为1,则计数变量加1
#         countnum = 0
#         for j in range(len(max_num_index_list)):
#             if(labels[i][max_num_index_list[j]] == 1.0):
#                 countnum += 1
#             else:
#                 continue
#         # 如果计数变量大等于10,我们认为该张图片预测成功
#         if(countnum >= 10):
#             pre_ok += 1
#
#     # 返回预测成功预batch大小的比值
#     return pre_ok/sampleNum
#
# # 得到抽取的名词
# def get_batch_NP(predict):
#     sampleNum = predict.shape[0]
#     batchExtractNP = []
#     batchPredValue = []
#     for i in range(sampleNum):
#         # 从predict[i]中找到前60大的索引列表
#         nums = list(predict[i])
#         max_num_index_list = list(map(nums.index, heapq.nlargest(60, nums)))
#         extractNP = []
#         extractNPValue = []
#         # 将得到的索引解析为名词短语
#         for j in range(len(max_num_index_list)):
#             extractNP.append(NPList50UKNP[max_num_index_list[j]])
#             extractNPValue.append(predict[i][max_num_index_list[j]])
#         batchExtractNP.append(extractNP)
#         batchPredValue.append(extractNPValue)
#     return batchExtractNP, batchPredValue


tf.reset_default_graph()
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

batch_size = 1
images, groundtruth_lists, _ = get_next_batch(batch_size)

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))


stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'test2'
request.model_spec.signature_name = 'predict_images'
request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(images, shape=[batch_size, 640, 640, 3], dtype=tf.float32))
#request.outputs['inputy'].CopyFrom(tf.contrib.util.make_tensor_proto(y, shape=[200, 1], dtype=tf.float32))
#.SerializeToString()
result = stub.Predict(request, 10.0) # 10 secs timeout
# result1 = stub.Regress(request, 10.0)
print (result)
