import tech_indicators
import yfinance as yf
from moomoo import *
import pandas as pd
import datetime as dt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from kerastuner.tuners import RandomSearch
import network
import numpy as np
import keras_tuner as kt
end_date = dt.date.today()
test_period = 10
train_period  = 49
test_date = end_date-dt.timedelta(days = test_period)
train_date = test_date-dt.timedelta(days = train_period)
stocklist = ['TSLA']
def get_stock_price(stocklist,start_date,end_date,interval):
    stockprice = {}
    for stock in stocklist:
        if not isinstance(stock, float):
            stock_data = yf.download(stock,start_date,end_date,interval = interval)
            stockprice[stock] = stock_data[['Close','High','Low']]
    return stockprice
test_stock_price =  get_stock_price(stocklist,test_date,end_date,'15m')
train_stock_price = get_stock_price(stocklist,train_date,test_date,'15m')
techA = tech_indicators.tech_indicators(train_stock_price,'TSLA')
techlist = ['kalmanfilter','stochR','ADX','ADXsignal','ATRsignal']
x_train = pd.DataFrame(columns = ['kalmanfilter','stochR','ADX','ADXsignal','ATRsignal'])
x_train['kalmanfilter'] = techA.kalmanfilter(14)
x_train['stochR'] = techA.stoch_R(14,3)
x_train['ADX'] = techA.ADX(14)[0]
x_train['ADXsignal'] = techA.ADX(14)[1]
x_train['ATRsignal'] = techA.ATR(14)[1]
x_train =x_train.dropna(axis = 0)
train_stock_return = np.log(train_stock_price['TSLA']['Close']/train_stock_price['TSLA']['Close'].shift(1))
upper = train_stock_return.mean()+0.67*train_stock_return.std()
lower = train_stock_return.mean()-0.67*train_stock_return.std()
signal = train_stock_return.apply(lambda x: 1 if x > upper else -1 if x < lower else 0)
y_train = signal.reindex(x_train.index)
y_train = y_train+1
y_train = keras.utils.to_categorical(y_train, num_classes=3)
numerical_features = ['kalmanfilter','stochR','ADX']
scaler = StandardScaler()
x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
#######################################################
#model train
#######################################################

# 初始化 Tuner
newnetwork = network.network()


tuner = RandomSearch(
    newnetwork,
    objective='val_accuracy',  # 优化的目标
    max_trials=10,             # 尝试 10 次不同的超参数组合
    executions_per_trial=1,    # 每个超参数组合运行的次数
    directory='my_dir',        # 保存结果的目录
    project_name='stock_signals'  # 项目名称
)
print(tuner.search_space_summary())
# 搜索批次大小和 epochs
tuner.search(
    x_train,
    y_train,
    epochs = 50,
    Batch_size = 64,
    verbose = 1
     ) # 动态调整 epochs


# 打印最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
最佳超参数：
- 层1神经元数量: {best_hps.get('units_layer1')}
- 层1 Dropout: {best_hps.get('dropout_layer1')}
- 层2神经元数量: {best_hps.get('units_layer2')}
- 层2 Dropout: {best_hps.get('dropout_layer2')}
- 学习率: {best_hps.get('learning_rate')}
""")
best_model = tuner.hypermodel.build(best_hps)

# 在验证集上评估
eval_result = best_model.evaluate(x_train, y_train, verbose=1)
print(f"最佳模型的评估结果 - 验证损失: {eval_result[0]}, 验证准确率: {eval_result[1]}")
import matplotlib.pyplot as plt

# 使用最佳超参数重新训练模型并记录训练过程
history = best_model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,  # 使用 20% 数据作为验证集
    verbose=1
)

# 绘制训练和验证的损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 绘制训练和验证的准确率
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()