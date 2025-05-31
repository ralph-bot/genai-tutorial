from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

# 加载加利福尼亚房价数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# 转换为DataFrame便于查看
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 中位数房价（目标变量）

print("数据基本信息：",df.columns)

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 1000:
    # 小数据集（行数少于1000）查看全量统计信息
    print("数据全部内容统计信息：")
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 大数据集查看数据前几行信息
    print("数据前几行内容统计信息：")
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 特征和目标变量
X = housing.data
y = housing.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
## 定义要比较的模型
models = {
    "线性回归": LinearRegression(),
    "Ridge回归": Ridge(alpha=1.0),
    "Lasso回归": Lasso(alpha=0.1),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
    "GBDT":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42,max_depth=3)
}

# 训练并评估模型
results = {}
for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # 评估指标
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    

#R²=1：模型完美拟合数据，所有预测值与实际值完全一致。
#R²=0：模型预测效果等同于使用均值进行预测（即模型没有捕捉到任何数据中的模式）。
#R²<0：模型拟合效果比使用均值预测还差（这种情况通常表示模型严重欠拟合）。
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'model': model
    }
    
    print(f"\n{name}：")
    print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

