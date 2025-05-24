import backtrader as bt 
import pandas as pd
import datetime
import sys 
import os 

class DataLoader:
    def __init__(self):
        pass
    def load_data_from_csv(self,data_path,start_date="2024-01-01",end_date="2024-12-31"):
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        data.sort_index(inplace=True)  # 按时间排序
        # 取出指定时间段的数据
        data = data.loc[start_date:end_date]
        # 重置索引
        #data.reset_index(inplace=True)
        print("-----\n",data.head())
        bt_data =  bt.feeds.PandasData(
            dataname=data,
            open='open',    # 映射开盘价列
            high='high',    # 映射最高价列
            low='low',      # 映射最低价列
            close='close',  # 映射收盘价列
            volume='volume' # 映射成交量列
        )
        return bt_data
    
    def load_data_from_dir(self,cerebro,data_dir,start_date="2024-01-01",end_date="2024-12-31"):
        #data_list = []
        for filename in os.listdir(data_dir):
            code = filename.split('.')[0]
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                bt_data = self.load_data_from_csv(data_path,start_date,end_date)
                cerebro.adddata(bt_data,name=code)
        return cerebro
    
    def load_data_from_file(self,cerebro,filename,start_date="2024-01-01",end_date="2024-12-31"):
        code = filename.split('.')[0]
        bt_data = self.load_data_from_csv(filename,start_date,end_date)
        cerebro.adddata(bt_data,name=code)
        return cerebro

