# coding = "utf-8"

import tkinter as tk
import threading
import time
from dataVisualization import tendPieChart
from sohuSpider import version1_0
from sentimentRecognize import GetRecognizeResult
from modelDisplay import catModel


def visual(text1,text2):
    t = threading.Thread(target=tendPieChart.picVisual, args=(text1,text2,))
    t.setDaemon(True)
    t.start()

def crawl(text1,text2):
    t = threading.Thread(target=version1_0.crawlThread,args=(text1,text2,))
    t.setDaemon(True)
    t.start()

def sentAnay(text1,text2):
    t = threading.Thread(target=GetRecognizeResult.SentAnayThread, args=(text1, text2,))
    t.setDaemon(True)
    t.start()

def displayModel():
    t = threading.Thread(target=catModel.displayModelThread())
    t.setDaemon(False)
    t.start()

def displayWordDict(text2):
    text2.insert(1.0, "C://PyWorkSpace/FangChuan/ruleModel/resources" + "\n")

if __name__=="__main__":
    top = tk.Tk()
    top.geometry('800x500+300+100')
    label = tk.Label(top,text='中文舆情分析系统',font=('Helvetica',18,'bold'))
    label.pack()

    label1 = tk.Label(top, text='输入:', font=('Helvetica', 12))
    label1.place(x=10,y=50,anchor='nw')

    text1 = tk.Text(top,width=110,height=10)
    text1.place(x=10,y=80,anchor='nw')

    label2 = tk.Label(top, text='显示:', font=('Helvetica', 12))
    label2.place(x=10, y=230, anchor='nw')

    text2 = tk.Text(top, width=110, height=10)
    text2.place(x=10, y=260, anchor='nw')


    button1 = tk.Button(top,text='抓取',width=10,height=1,command = lambda:crawl(text1,text2))
    button1.place(x=10,y=420,anchor='nw')

    button2 = tk.Button(top, text='识别', width=10, height=1,command = lambda:sentAnay(text1,text2))
    button2.place(x=125, y=420, anchor='nw')

    button3 = tk.Button(top, text='可视化', width=10, height=1, command=lambda: tendPieChart.picVisual(text1, text2))
    button3.place(x=240, y=420, anchor='nw')

    button4 = tk.Button(top, text='词典展示', width=10, height=1, command=lambda: displayWordDict(text2))
    button4.place(x=355, y=420, anchor='nw')

    button5 = tk.Button(top, text='模型展示', width=10, height=1, command=lambda:displayModel())
    button5.place(x=470, y=420, anchor='nw')

    tk.mainloop()