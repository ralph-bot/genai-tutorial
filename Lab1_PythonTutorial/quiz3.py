import sys
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np

# 画出成交量的柱状图
def get_volume_and_draw(filename):
    # pandas 读取数据
    print("hello")
    df = pd.read_csv(filename)
    print(df)
    # 计算成交量
    df['volume'] 
    # 画出成交量的柱状图
    sns.barplot(x=df['datetime'], y=df['volume'])

    plt.show()

# 统计上涨的天数和下跌的天数， 并且用直方图表示。
def cal_up_and_down(filename):
    # pandas 读取数据
    df = pd.read_csv(filename)
    # 计算涨跌幅
    df['change'] = (df['close'] - df['open']) / df['open']
    df['change'] = np.where(df['change'] > 0, 1, 0)
    # 统计涨跌幅
    sns.displot(df['change'])
    plt.show()
    
#cal_up_and_down("000001.csv")
get_volume_and_draw("000001.csv")




