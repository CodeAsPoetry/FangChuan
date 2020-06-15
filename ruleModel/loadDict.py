# coding = 'utf-8'

# 加载否定词典
nonPath = 'resources/nonWord.txt'
nonList = []
for line in open(nonPath, 'r', encoding='utf-8'):
    nonList.append(line.strip().replace('\n', ''))
print('nonList', len(nonList), nonList)

# 加载李军正向情感词典
posPath = 'resources/tsinghua.positive.gb.txt'
posList = []
for line in open(posPath, 'r', encoding='utf-8'):
    posList.append(line.strip().replace('\n', ''))
print('posList', len(posList), posList)

# 加载李军负向情感词典
negPath = 'resources/tsinghua.negative.gb.txt'
negList = []
for line in open(negPath, 'r', encoding='utf-8'):
    negList.append(line.strip().replace('\n', ''))
print('negList', len(negList), negList)

# 加载自定义扩充正向情感词典
definePosPath = 'resources/definePos.txt'
definePosList = []
for line in open(definePosPath, 'r', encoding='utf-8'):
    definePosList.append(line.strip().replace('\n', ''))
print('definePosList', len(definePosList), definePosList)

# 加载自定义扩充负向情感词典
defineNegPath = 'resources/defineNeg.txt'
defineNegList = []
for line in open(defineNegPath, 'r', encoding='utf-8'):
    defineNegList.append(line.strip().replace('\n', ''))
print('defineNegList', len(defineNegList), defineNegList)

# 加载程度副词词典
degreePath = 'resources/degreeWord.txt'
degreeList = []
for line in open(degreePath, 'r', encoding='utf-8'):
    if '|' not in line:
        degreeList.append(line.strip().replace('\n', ''))
print('degreeList', len(degreeList), degreeList)


# 规则：
# 1. 出现一个正向情感词加1，出现一个负向情感词加-1。
# 2. 程度副词在前，否定词在后，情感倾向反转；否定词在前，程度副词在后，情感倾向不变。
# 3. 单个(奇数)否定，情感倾向反转；双重否定(偶数)表肯定，情感倾向不变。

import jieba

# 太耿直，踩着点退休了
# 他的心态不是不健康
# 这个书包不是很好
# 这个书包很是不好
# 人才前进的道路就这样粗暴扼杀。
# 兄弟，你可真秀
# 给老子把意大利炮拉上来
# 头上的绿帽子真好看


sentence = '这个书包相当不精美'
senList = list(jieba.cut(sentence))
print('senList', len(senList), senList)

nonWordInSent = []
posWordInSent = []
negWordInSent = []
degreeInSent = []

for i_index in range(len(senList)):
    if senList[i_index] in nonList:
        nonWordInSent.append(senList[i_index])
    if senList[i_index] in degreeList:
        degreeInSent.append(senList[i_index])
    if senList[i_index] in negList:
        negWordInSent.append(senList[i_index])
    if senList[i_index] in defineNegList:
        negWordInSent.append(senList[i_index])
    if senList[i_index] in posList:
        posWordInSent.append(senList[i_index])
    if senList[i_index] in definePosList:
        posWordInSent.append(senList[i_index])

# for item in nonList:
#     if item in sentence:
#         nonWordInSent.append(item)

print('nonWordInSent', len(nonWordInSent), nonWordInSent)
print('posWordInSent', len(posWordInSent), posWordInSent)
print('negWordInSent', len(negWordInSent), negWordInSent)
print('degreeInSent', len(degreeInSent), degreeInSent)

for item in definePosList:
    if item in sentence:
        posWordInSent.append(item)

for item in defineNegList:
    if item in sentence:
        negWordInSent.append(item)


count = 0
print(len(negWordInSent))
if len(negWordInSent) > 1:
    for item in negWordInSent:
        if item in nonWordInSent:
            count += 1

print(count)
print(len(posWordInSent))
print(len(negWordInSent)-count)
score = len(posWordInSent) - (len(negWordInSent)-count)
print(score)
# 双重否定表肯定
if len(nonList) %2 != 0 and len(degreeList) == 0:
    score = -score

if len(nonList) %2 != 1 and len(degreeList) > 0:
    degreeIndex = senList.index(degreeList[0])
    nonIndex = senList.index(nonList[0])
    if nonIndex < degreeIndex:
        if score < 0:
            score = -score
    else:
        if score > 0:
            score = -score

print(score)
if score <= 0:
    print('负向')
else:
    if score>0:
        print('正向')










