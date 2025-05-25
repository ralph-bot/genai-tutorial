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
    df = pd.read_csv(filename)
    print(df)


# 统计上涨的天数和下跌的天数， 并且用直方图表示。
def cal_up_and_down(filename):
    # pandas 读取数据
    df = pd.read_csv(filename)

    plt.show()
    
#cal_up_and_down("000001.csv")
get_volume_and_draw("Lab2/000001.csv")




