import backtrader as bt
import pandas as pd
import datetime
import BasicStrategy
import MoveAverage as qs
import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

def config_cerebro(cerebro):
    # 添加分析指标
    # 返回年初至年末的年度收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    # 计算最大回撤相关指标
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    # 计算年化收益：日度收益
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns', tann=252)
    # 计算年化夏普比率：日度收益
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0) # 计算夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    # 启动回测
    return cerebro

def eval_strategy(cerebro,result):
    strat = result[0]
        # 获取策略的收益率序列
    returns = pd.Series(strat.analyzers._TimeReturn.get_analysis())
    print("returns: ", returns)
    # 打印评价指标
    print("--------------- AnnualReturn -----------------")
    print(strat.analyzers._AnnualReturn.get_analysis())
    print("--------------- DrawDown -----------------") 
    print(strat.analyzers._DrawDown.get_analysis())
    print("--------------- Returns -----------------")
    print(strat.analyzers._Returns.get_analysis())
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers._SharpeRatio.get_analysis())
    print("--------------- SharpeRatio_A -----------------")
    print(strat.analyzers._SharpeRatio_A.get_analysis())
        
    # 常用指标提取
    analyzer = {}
    # 提取年化收益
    analyzer['年化收益率'] = result[0].analyzers._Returns.get_analysis()['rnorm']
    analyzer['年化收益率（%）'] = result[0].analyzers._Returns.get_analysis()['rnorm100']
    # 提取最大回撤
    analyzer['最大回撤（%）'] = result[0].analyzers._DrawDown.get_analysis()['max']['drawdown'] * (-1)
    # 提取夏普比率
    analyzer['年化夏普比率'] = result[0].analyzers._SharpeRatio_A.get_analysis()['sharperatio']
    print("--------------- 常用指标 -----------------")
    print(analyzer)



if __name__ == '__main__':
    # 实例化 cerebro
    cerebro = bt.Cerebro()
    #daily_price = pd.read_csv("backtrader/learn_backtrader-master/Data/daily_price.csv", parse_dates=['datetime'])
    # 将策略添加给大脑
    print(qs.__file__)
    # add strategy to cerebro
    cerebro.addstrategy(qs.MovingAverageStrategy)
    # 加载数据
    dl = DataLoader.DataLoader() 
    #csv_path  = "/Users/fengrao/Workspace/3_NextWork/1_stock/stock_analyze/data/trade_info/002316.csv"
    #data = dl.load_data_from_csv(csv_path)
    #cerebro = dl.load_data_from_dir(cerebro,csv_dir)
    file_dir=os.path.dirname(os.path.abspath(__file__))
    cerebro = dl.load_data_from_file(cerebro,os.path.abspath(file_dir+"/data/000001.csv"))
    # if you want to add date range, you can use the following code
    #cerebro = dl.load_data_from_dir(cerebro,csv_dir,"2020-01-01","2024-04-01")  



    # 将数据添加给大脑
    #cerebro.adddata(data,name="002316")   
    # 配置策略参数
    cerebro = config_cerebro(cerebro)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    result = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    #cerebro.plot()  # and plot it with a single command
    print("result: ", result)

    # calcuate the alpha beta
    # 从返回的 result 中提取回测结果
    #strat = result[0]

    eval_strategy(cerebro,result)
    cerebro.plot() 
    

