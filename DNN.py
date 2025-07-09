import network
from kerastuner.tuners import RandomSearch
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # Split dataset into train and test
from sklearn.preprocessing import OneHotEncoder, PowerTransformer  # Encoding & scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer  # Apply transformations to specific columns
from sklearn.ensemble import RandomForestClassifier  # Random Forest model for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Model evaluation metrics
from sklearn.utils import resample  # Resampling methods for balancing data
import pandas as pd

data= pd.read_csv(r'C:\Users\user\Desktop\equipment_anomaly_data.csv')
X= data.drop('faulty', axis=1)
y= data['faulty']
# Numeric and string features
numeric_features= X.select_dtypes(exclude=['object']).columns
string_features= X.select_dtypes(include=['object']).columns
normal_cols = ['temperature', 'pressure', 'vibration', 'humidity']

for col in normal_cols:

    upper_limit = data[col].mean() + (3 * data[col].std())
    lower_limit = data[col].mean() - (3 * data[col].std())

    data[col] = np.where(
        data[col] > upper_limit,
        upper_limit,

        np.where(
            data[col] < lower_limit,
            lower_limit,
            data[col]
        )
    )
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
x = pd.get_dummies(X[string_features],dtype = int)
X_process = pd.concat([X[numeric_features],x],axis = 1)
X_process = X_process.dropna(axis = 1)

def cross_validation_group(X, y, train_idx, test_idx):
    x_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    x_val, y_val = X.iloc[test_idx, :], y.iloc[test_idx]
    return x_train, y_train, x_val, y_val

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
    X_train,
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
eval_result = best_model.evaluate(X_test, y_test, verbose=1)
print(f"最佳模型的评估结果 - 验证损失: {eval_result[0]}, 验证准确率: {eval_result[1]}")
import matplotlib.pyplot as plt

# 使用最佳超参数重新训练模型并记录训练过程
history = best_model.fit(
    X_test,
    y_test,
    epochs=60,
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