from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

from generate_pips import FOLDER, SEED
from utils import plot_cm, plot_metrics, plot_roc

def main():
    print(f'{datetime.now()} Loading data..')
    X = pd.read_hdf(FOLDER / 'preprocessed.h5').values
    y = pd.read_csv(FOLDER / 'labels.csv')['action'].values
    print(f'{datetime.now()} Loaded successfully.')
    print(f'{datetime.now()} Started shuffling..')
    X, y = shuffle(X, y, random_state=SEED)
    print(f'{datetime.now()} Finished shuffling.')

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
        keras.Input(shape=(22447, ), name='inputs'),
        keras.layers.Dense(
            5000,
            input_shape=(22447, ),
            activation=tf.nn.relu,
            name='dense_1'),
        keras.layers.Dense(
            5000,
            input_shape=(5000, ),
            activation=tf.nn.relu,
            name='dense_2'),
        keras.layers.Dense(
            1,
            input_shape=(5000, ),
            activation=tf.nn.sigmoid,
            name='output'),
    ])

    print(model.summary())

    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
    )
    keras.utils.plot_model(model, FOLDER / 'model.png', show_shapes=True, rankdir='LR')
    print(f'\n\n\n Model compiled.\n\n\n')
    history = model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=50,
        verbose=2,
        validation_data=(X_val, y_val)
        )

    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(len(y_pred))

    plot_metrics(
        history,
        optimizer='ADAM',
        loss='BCE',
        figdir=FOLDER / 'history.png',
        show=True,
    )
    plot_roc(
        y_test,
        y_pred,
        FOLDER / 'roc.png',
        '22447/5000/5000/1',
    )
    plot_cm(
        y_test,
        y_pred > 0.5,
        FOLDER / 'cm.png',
        '22447/5000/5000/1',
        p=0.5,
    )
    plt.show()


if __name__ == '__main__':
    main()
