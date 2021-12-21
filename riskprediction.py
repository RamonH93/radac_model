import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tensorflow import keras

SEED = 1

df = pd.read_csv('riskdataset.csv')

y = df.pop('secrisk')

X = df.values
y = y.values

X, y = shuffle(X, y, random_state=SEED)

test_size = 0.2
data_len = len(y)
split_len = int(test_size * data_len)
train_split_idx = data_len - 2 * split_len
test_split_idx = data_len - split_len
X_train, y_train = X[:train_split_idx], y[:train_split_idx]
X_val, y_val = X[train_split_idx:test_split_idx], y[
    train_split_idx:test_split_idx]
X_test, y_test = X[test_split_idx:], y[test_split_idx:]

model = keras.models.Sequential([
    keras.Input(shape=(X_train.shape[1], ), name='inputs'),
    keras.layers.Dense(
        2,
        input_shape=(X_train.shape[1], ),
        activation=tf.nn.relu,
        name='dense_1'),
    # keras.layers.Dense(
    #     50,
    #     input_shape=(X_train.shape[1], ),
    #     activation=tf.nn.relu,
    #     name='dense_2'),
    keras.layers.Dense(
        1,
        input_shape=(2, ),
        activation=tf.nn.sigmoid,
        name='output'),
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.2),
    loss='mean_squared_error',
)

history = model.fit(X_train, y_train, epochs=1000, verbose=2, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)
y_pred = y_pred.reshape(len(y_pred))
print(y_pred)
rmse = mean_squared_error(y_pred, y_test, squared=False)
print(rmse)

keras.utils.plot_model(model, show_shapes=True, rankdir='LR')

from utils import plot_metrics
plot_metrics(history, optimizer='SGD', loss='MSE', show=True)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
# ax.set_xlim(0, max(y_pred)*(1+2*rmse))
# ax.set_ylim(0, max(y_pred)*(1+2*rmse))
# ax.get_xlim()
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted Security Risk')
plt.show()
