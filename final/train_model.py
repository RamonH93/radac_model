from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from generate_pips import FOLDER
from utils import plot_cm, plot_metrics, plot_roc

def main(risk=False):
    print(f'{datetime.now()} Loading data..')
    # X = pd.read_hdf(FOLDER / 'preprocessed.h5').values
    # y = pd.read_csv(FOLDER / 'labels.csv')['action'].values
    npzfile = np.load(FOLDER / 'Xys.npz')
    X = npzfile['X']
    if risk:
        y = npzfile['y_r']
    else:
        y = npzfile['y_a']
    print(f'{datetime.now()} Loaded successfully.')

    test_size = 0.2
    data_len = len(y)
    split_len = int(test_size * data_len)
    train_split_idx = data_len - 2 * split_len
    test_split_idx = data_len - split_len
    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:test_split_idx], y[
        train_split_idx:test_split_idx]
    X_test, y_test = X[test_split_idx:], y[test_split_idx:]

    if risk:
        model = keras.models.Sequential([
            keras.Input(shape=(X_train.shape[1], ), name='inputs'),
            keras.layers.Dense(
                100,
                input_shape=(X_train.shape[1], ),
                activation=tf.nn.relu,
                name='dense_1'),
            # keras.layers.Dense(
            #     100,
            #     input_shape=(3, ),
            #     activation=tf.nn.relu,
            #     name='dense_2'),
            keras.layers.Dense(
                1,
                input_shape=(100, ),
                activation=keras.activations.linear,
                name='output')
        ])
    else:
        model = keras.models.Sequential([
        keras.Input(shape=(X_train.shape[1], ), name='inputs'),
        keras.layers.Dense(
            3,
            input_shape=(X_train.shape[1], ),
            activation=tf.nn.relu,
            name='dense_1'),
        # keras.layers.Dense(
        #     100,
        #     input_shape=(3, ),
        #     activation=tf.nn.relu,
        #     name='dense_2'),
        keras.layers.Dense(
            1,
            input_shape=(X_train.shape[1], ),
            activation=tf.nn.sigmoid,
            name='output')
        ])

    print(model.summary())

    if risk:
        loss = 'mean_squared_error'
    else:
        loss = 'binary_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(amsgrad=False),
        loss=loss,
    )
    keras.utils.plot_model(model, FOLDER / 'model.png', show_shapes=True, rankdir='LR')
    print(f'\n\n\n Model compiled.\n\n\n')
    history = model.fit(
        X_train, y_train,
        batch_size=4096,
        epochs=250,
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
        show=False,
    )
    if not risk:
        plot_roc(
            y_test,
            y_pred,
            FOLDER / 'roc.png',
            f'{datetime.now()}',
        )
        plot_cm(
            y_test,
            y_pred > 0.5,
            FOLDER / 'cm.png',
            f'{datetime.now()}',
            p=0.5,
        )


if __name__ == '__main__':
    main(True)
    
    # print(f'{datetime.now()} Loading data..')
    # X = pd.read_hdf(FOLDER / 'preprocessed.h5').values
    # y = pd.read_csv(FOLDER / 'labels.csv')['action'].values
    # npzfile = np.load(FOLDER / 'Xy.npz')
    # X = npzfile['X']
    # y = npzfile['y']
    # print(f'{datetime.now()} Loaded successfully.')
    # print(X[0])