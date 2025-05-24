import backtrader as bt
import datetime
import DataLoader
from BasicStrategy import *
# 定义一个简单的移动平均线策略
'''
移动平均线策略是一种常见的量化交易策略，它的基本思想是通过计算过去一段时间内的平均价格来判断市场的趋势。
如果当前价格高于过去一段时间内的平均价格，就认为市场处于上涨趋势，买入股票；如果当前价格低于过去一段时间内的平均价格，就认为市场处于下跌趋势，卖出股票。
移动平均线策略的实现通常包括以下几个步骤：
1. 收集历史数据：收集过去一段时间内的股票价格数据。
2. 计算移动平均线：计算过去一段时间内的平均价格。
3. 判断趋势：根据当前价格和过去一段时间内的平均价格，判断市场的趋势。
4. 执行交易：根据市场的趋势，执行买入或卖出操作。
5. 调整参数：根据市场的变化，调整移动平均线的周期和阈值。
'''

class MovingAverageStrategy(BasicStrategy):
    params = (
        ('ma_period', 5),  # 移动平均线的周期
        ('threshold', 0.01)  # 价格高于均线的阈值
    )

    def __init__(self):
        # 初始化用于存储过去五天的收盘价
        self.data_close = self.datas[0].close
        self.close_prices = []
    
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
    def next(self):
        # 将当前收盘价添加到列表中
        dt = self.datas[0].datetime.date(0)  # 获取当前的回测时间点 datas 是有多少个股票，即股票的个数== len(datas)
        #print("--------------{} 当前回测时间----------".format(dt))
        self.close_prices.append(self.data_close[0])
        # 确保列表中只有过去五天的收盘价
        if len(self.close_prices) > self.params.ma_period:
            self.close_prices.pop(0)

        # 只有当收集到足够的价格数据时才进行交易判断
        if len(self.close_prices) == self.params.ma_period:
            # 计算过去五天的平均价格
            MA5 = sum(self.close_prices) / self.params.ma_period
            # 获取当前价格
            current_price = self.data_close[0]
            # 获取当前可用现金
            cash = self.broker.getcash()
            self.log(f"当前可用现金: {cash} 当前价格: {current_price} 五天平均价: {MA5}")
            # 如果上一时间点价格高出五天平均价1%，则全仓买入
            if current_price > (1 + self.params.threshold) * MA5 and cash > 0:
                # 计算可以买入的股票数量
                size = cash / current_price
                size = int(size/100)*100  # A 股特有的手的概念， 100股 = 1手
                if size > 0:
                    # 记录这次买入
                    self.log(f"价格高于均价 {self.params.threshold * 100}%, 买入 {self.datas[0]._name} , 买入数量: {size}")
                    # 用所有现金买入股票
                    self.buy(size=size,data=self.datas[0],price = current_price )
            # 如果上一时间点价格低于五天平均价，则空仓卖出
            elif current_price < MA5 and self.getposition(self.datas[0]).size > 0:
                # 记录这次卖出
                self.log(f"价格低于均价, 卖出 {self.datas[0]._name}, 卖出数量: {self.getposition(self.datas[0]).size}")
                # 卖出所有股票
                self.sell(size=self.getposition(self.datas[0]).size,data=self.datas[0],price = current_price)
                #self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')



if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(MovingAverageStrategy)

    # 加载数据
    dl = DataLoader.DataLoader() 
    #csv_path  = "/Users/fengrao/Workspace/3_NextWork/1_stock/stock_analyze/data/trade_info/002316.csv"
    csv_dir = "/Users/fengrao/Workspace/3_NextWork/1_stock/stock_analyze/data/toy"
    #data = dl.load_data_from_csv(csv_path)
    #cerebro = dl.load_data_from_dir(cerebro,csv_dir,"2020-01-01","2024-04-01")
    cerebro = dl.load_data_from_dir(cerebro,csv_dir)

    # 设置初始资金
    cerebro.broker.setcash(100000.0)


       # 加载数据，设置开始时间和结束时间
    # data = bt.feeds.YahooFinanceData(
    #     dataname='AAPL',
    #     fromdate=datetime.datetime(2023, 1, 1),
    #     todate=datetime.datetime(2023, 12, 31)
    # )
    # # 将数据添加到 Cerebro 引擎
    # cerebro.adddata(data)


    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # 运行回测
    cerebro.run()
    cerebro.plot() 
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    