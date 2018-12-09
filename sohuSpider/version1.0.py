# coding = "utf-8"

from selenium import webdriver
import time

# 原生未标注语料目录
nativeCorpusDirPath = "F://FangChuan/NativeCorpus/"
# 存放所有评论
comment=[]
# 启动浏览器
browser = webdriver.Chrome()
# 打开要抓取的新闻页面，修改搜狐新闻文件时，需将搜狐新闻文件的网页地址填入即可
browser.get('http://www.sohu.com/a/280581674_391294?_f=index_chan08news_15')
# 根据XPath获取评论总数
comment_num_element =browser.find_element_by_xpath("//a[@href='#comment_area']/span[@class='num']")
comment_num = (int)(comment_num_element.text)
# 根据XPath获取新闻标题文本
newTitleElement = browser.find_element_by_xpath("//div[@class='text-title']/h1")
newTitle = newTitleElement.text
# 以新闻标题为将要保存到的文件名
nativeCorpusFilePath = nativeCorpusDirPath+newTitle+".txt"
# 打开文件
nativeCorpusFile = open(nativeCorpusFilePath,"w",encoding="utf-8")
# 根据XPath获取当前页的评论元素集合
comment_current_elements = browser.find_elements_by_xpath('//div[@class="c-discuss"]')
# 遍历当前页评论集合
for i_comment in comment_current_elements:
    print(i_comment.text)
    comment.append(i_comment.text)
# 计数变量
i=0
# 循环获取评论
while(i<=comment_num):
    # 获取跳到下页的链接元素
    isGet = browser.find_element_by_xpath('//div[@class="c-comment-more"]')
    # 点击事件，获取后续评论
    isGet.click()
    time.sleep(3)
    comment_next_elements = browser.find_elements_by_xpath('//div[@class="c-discuss"]')
    for i_comment in comment_next_elements:
        print(i_comment.text)
        comment.append(i_comment.text)
        nativeCorpusFile.writelines(i_comment.text)
        nativeCorpusFile.write("\n")
        i+=1
# 关闭文件及浏览器
nativeCorpusFile.close()
browser.close()



