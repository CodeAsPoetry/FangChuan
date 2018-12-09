# coding = "utf-8"
import matplotlib.pyplot as plot
import os

# 标注语料文件目录
labeledCorpusDirPath = "F://FangChuan/labeledCorpus/"
# 获取标注语料目录下的所有文件名，同时将每个文件名保存到title集合中，以便作图时生成标题
childLabeledFilePath=[]
title=[]
parents = os.listdir(labeledCorpusDirPath)
for parent in parents:
    title.append(parent)
    labeledChild = os.path.join(labeledCorpusDirPath,parent)
    childLabeledFilePath.append(labeledChild)

# 修改不同新闻文件，需改动childLabeledFilePath[0]中的序号
labeledFile = open(childLabeledFilePath[0],"r",encoding="utf-8")

# 分别统计正面、中性、负面的评论数
posNum=0
midNum=0
negNum=0
text = labeledFile.readline().strip("\n")
while(text !=''):
    label = int(text.split(" ")[1][-1])
    if label==0:
        negNum+=1
    elif label==1:
        midNum+=1
    else:
        posNum+=1
    text = labeledFile.readline().strip("\n")
# 生成每个部分所占比例
posPred = posNum/(posNum+midNum+negNum)
midPred = midNum/(posNum+midNum+negNum)
negPred = negNum/(posNum+midNum+negNum)

# 饼状图
labels = u'Positive', u'neutral', u'Negative'
fracs = [posPred, midPred, negPred]
explode = [0.1, 0, 0]  # 0.1 凸出这部分，
plot.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse

# 保证中文在饼状图中不乱码
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
# 制表题
plot.title(title[0][0:-4],fontproperties=font)
patches, l_text, p_text = plot.pie(fracs, explode=explode, labels=labels,
                                   labeldistance=1.1, autopct='%3.2f%%', shadow=True,
                                   startangle=90, pctdistance=0.6)

# labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
# autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
# shadow，饼是否有阴影
# startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
# pctdistance，百分比的text离圆心的距离
# patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本
# # 改变文本的大小
# # 方法是把每一个text遍历。调用set_size方法设置它的属性
# for t in l_text:
#     t.set_size = 30
# for t in p_text:
#     t.set_size = 10

plot.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.0))
# loc: 表示legend的位置，包括'upper right','upper left','lower right','lower left'等
# bbox_to_anchor: 表示legend距离图形之间的距离，当出现图形与legend重叠时，可使用bbox_to_anchor进行调整legend的位置
# 由两个参数决定，第一个参数为legend距离左边的距离，第二个参数为距离下面的距离
plot.grid()
plot.show()
