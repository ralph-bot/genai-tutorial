import sys
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt 
import os

'''
统计Top10 天数的成交量
'''
with open("000001.csv") as fp : 
    # Read the CSV file and skip the header
    next(fp)  # Skip header line
    # add your code here