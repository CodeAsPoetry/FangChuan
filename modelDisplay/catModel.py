# coding = "utf-8"

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def displayModelThread():
    try:
        chrome_options = Options()
        # 启动浏览器
        browser = webdriver.Chrome(chrome_options=chrome_options)
        # 打开要抓取的新闻页面，修改搜狐新闻文件时，需将搜狐新闻文件的网页地址填入即可
        target_url = 'localhost:6006'
        print(target_url)
        browser.get(target_url)
        # while True:
        #     time.sleep(1000)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    displayModel()



