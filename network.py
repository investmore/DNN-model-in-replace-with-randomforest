from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import keras_tuner as kt
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
class network(kt.HyperModel):
    def build(self,hp,*args, **kwargs):
        model = Sequential()
        model.add(Dense(
            units=hp.Int('units_layer1', min_value=32, max_value=128, step=32),  # 层1的神经元数量
            activation='relu')
        )
        model.add(Dropout(hp.Float('dropout_layer1', min_value=0.2, max_value=0.5, step=0.1)))  # Dropout
        model.add(Dense(
            units=hp.Int('units_layer2', min_value=32, max_value=128, step=32),  # 层2的神经元数量
            activation='relu'
        ))
        model.add(Dropout(hp.Float('dropout_layer2', min_value=0.2, max_value=0.5, step=0.1)))  # Dropout
        model.add(Dense(1, activation='sigmoid'))  # 输出层
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),  # 学习率选择
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )