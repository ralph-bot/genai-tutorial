import backtrader as bt
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
import DataLoader 
import os
import backtrader.indicators as btind # 导入策略分析模块



class BasicStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )
  
    def log(self, txt):
        '''Logging function'''
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')
    
    # 初始化函数
    def __init__(self):
        '''初始化属性、计算指标等'''
        # 指标计算可参考《backtrader指标篇》
        #self.add_timer() # 添加定时器
        print("init basic strategy")
        pass

    # 整个回测周期上，不同时间段对应的函数
    def start(self):
        '''在回测开始之前调用,对应第0根bar'''
        # 回测开始之前的有关处理逻辑可以写在这里
        # 默认调用空的 start() 函数，用于启动回测
        pass
    
    def prenext(self):
        '''策略准备阶段,对应第1根bar ~ 第 min_period-1 根bar'''
        # 该函数主要用于等待指标计算，指标计算完成前都会默认调用prenext()空函数
        # min_period 就是 __init__ 中计算完成所有指标的第1个值所需的最小时间段
        pass
    
    def nextstart(self):
        '''策略正常运行的第一个时点，对应第 min_period 根bar'''
        # 只有在 __init__ 中所有指标都有值可用的情况下，才会开始运行策略
        # nextstart()只运行一次，主要用于告知后面可以开始启动 next() 了
        # nextstart()的默认实现是简单地调用next(),所以next中的策略逻辑从第 min_period根bar就已经开始执行
        pass
    
    
    def next(self):
        '''策略正常运行阶段，对应第min_period+1根bar ~ 最后一根bar'''
        # 主要的策略逻辑都是写在该函数下
        pass
    
    def stop(self):
        '''策略结束，对应最后一根bar'''
        # 告知系统回测已完成，可以进行策略重置和回测结果整理了
        pass
    
   
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受，不做处理
            return

        if order.status == order.Completed:
            if order.isbuy():
                print('买入交易成功')
            elif order.issell():
                print('卖出交易成功')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('交易失败，订单状态：', order.Status[order.status])

        # 重置订单状态
        self.order = None
    def notify_trade(self, trade):
        '''通知交易信息'''
        pass
    
    def notify_cashvalue(self, cash, value):
        '''通知当前资金和总资产'''
        pass
    
    def notify_fund(self, cash, value, fundvalue, shares):
        '''返回当前资金、总资产、基金价值、基金份额'''
        pass

    def notify_store(self, msg, *args, **kwargs):
        '''返回供应商发出的信息通知'''
        pass
    
    def notify_data(self, data, status, *args, **kwargs):
        '''返回数据相关的通知'''
        pass
    
    def notify_timer(self, timer, when, *args, **kwargs):
        '''返回定时器的通知'''
        # 定时器可以通过函数add_time()添加
        pass

