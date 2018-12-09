# coding= "utf-8"

from aip import AipNlp
import os
import time

# 标注语料的文件目录
labeledCorpusDirPath = "F://FangChuan/labeledCorpus/"
# 原生语料的文件目录
nativeCorpusDirPath = "F://FangChuan/NativeCorpus/"


childNativeFilePath=[]
childLabeledFilePath=[]
# 获取原生语料目录下的所有文本文件，同时在标注目录下生成对应文件路径
parents = os.listdir(nativeCorpusDirPath)
for parent in parents:
    nativeChild = os.path.join(nativeCorpusDirPath,parent)
    childNativeFilePath.append(nativeChild)
    labeledChild = os.path.join(labeledCorpusDirPath,parent)
    childLabeledFilePath.append(labeledChild)

# 远程调用参数
APP_ID = 'XXXXXX'
API_KEY = 'XXXXXXXXXXXXX'
SECRET_KEY = 'XXXXXXXXXXXXXXXXXXXXXX'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

""" 0:负向，1:中性，2:正向"""
# 修改新闻文件是，需分别修改childNativeFilePath[0]和childLabeledFilePath[0]中的序号
nativeFile = open(childNativeFilePath[0],"r",encoding="utf-8")
labeledFile = open(childLabeledFilePath[0],"w",encoding="utf-8")

# 生成标注语料
id = 0
text = nativeFile.readline().strip("\n")
while(text !=''):
    id+=1
    try:
        text = text.encode("gbk")
        text = text.decode("gbk")
        result = client.sentimentClassify(text)
        print('id='+str(id)+" ",'label='+str(result['items'][0]['sentiment'])+" ",text)
        labeledFile.write('id='+str(id)+" ")
        labeledFile.write('label='+str(result['items'][0]['sentiment'])+" ")
        labeledFile.write(text)
        labeledFile.write("\n")
        time.sleep(1)
    except Exception as e:
        print(e)
    text = nativeFile.readline().strip("\n")

# 关闭资源
nativeFile.close()
labeledFile.close()


