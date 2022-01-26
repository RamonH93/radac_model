import os
from datetime import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from generate_pips import FOLDER, SEED
from utils import (
    MatthewsCorrelationCoefficient,
    plot_cm,
    plot_metrics,
    plot_roc,
)

EARLYSTOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=100,
    verbose=1
    )

def summarize_keras_trainable_variables(model, message):
    s = sum(map(lambda x: x.sum(), model.get_weights()))
    print("summary of trainable variables %s: %.13f" % (message, s))
    return s

def main(risk=False, regression=False):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.backend.clear_session()

    print(f'{datetime.now()} Loading data..')
    # X = pd.read_hdf(FOLDER / 'preprocessed.h5').values
    # y = pd.read_csv(FOLDER / 'labels.csv')['action'].values
    npzfile = np.load(FOLDER / 'Xys.npz')
    X = npzfile['X']
    if risk:
        y = npzfile['y_r']
        if not regression:
            y = pd.get_dummies(y.flatten()).values
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

    if risk and not regression:
        model = keras.models.Sequential([
            keras.Input(shape=(X_train.shape[1], ), name='inputs'),
            keras.layers.Dense(
                50,
                input_shape=(X_train.shape[1], ),
                activation=tf.nn.relu,
                name='dense_1'),
            keras.layers.Dense(
                100,
                input_shape=(50, ),
                activation=tf.nn.relu,
                name='dense_2'),
            keras.layers.Dense(
                100,
                input_shape=(100, ),
                activation=tf.nn.relu,
                name='dense_3'),
            keras.layers.Dense(
                len(y[0]),
                input_shape=(100, ),
                activation=keras.activations.softmax,
                name='output')
        ])
    elif risk:
        model = keras.models.Sequential([
            keras.Input(shape=(X_train.shape[1], ), name='inputs'),
            keras.layers.Dense(
                50,
                input_shape=(X_train.shape[1], ),
                activation=tf.nn.relu,
                name='dense_1'),
            keras.layers.Dense(
                100,
                input_shape=(50, ),
                activation=tf.nn.relu,
                name='dense_2'),
            keras.layers.Dense(
                100,
                input_shape=(100, ),
                activation=tf.nn.relu,
                name='dense_3'),
            keras.layers.Dense(
                len(y[0]),
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

    if risk and not regression:
        loss = 'categorical_crossentropy'
    elif risk:
        loss = 'mean_squared_error'
    else:
        loss = 'binary_crossentropy'

    keras.utils.get_custom_objects().update(
        {'MCC': MatthewsCorrelationCoefficient})

    if risk and not regression:
        metrics = ['categorical_accuracy']
    elif risk:
        metrics = [keras.metrics.RootMeanSquaredError()]
    else:
        metrics = [
            'accuracy',
            'AUC',
            'MCC',
        ]

    model.compile(
        optimizer=keras.optimizers.Adam(amsgrad=False),
        loss=loss,
        metrics=metrics,
    )
    summarize_keras_trainable_variables(model, "before training")
    # summary of trainable variables before training: -34.9238157272339
    keras.utils.plot_model(model, FOLDER / 'model.png', show_shapes=True, rankdir='LR')
    print(f'\n\n\n Model compiled.\n\n\n')

    history = model.fit(
        X_train, y_train,
        batch_size=4096,
        epochs=1000,
        verbose=2,
        callbacks=[],
        validation_data=(X_val, y_val)
        )

    summarize_keras_trainable_variables(model, "after training")
    # summary of trainable variables after training: -50.9813705242705

    y_pred = model.predict(X_test)#.flatten()
    # from sklearn.metrics import mean_squared_error
    # rmse = mean_squared_error( y_test, y_pred, squared=True)
    # print(np.count_nonzero(y_pred == y_test))

    plot_metrics(
        history,
        optimizer='ADAM',
        loss='BCE',
        figdir=FOLDER / 'history.png',
        show=False,
    )
    if risk and regression:
        plt.plot([0, 1], [0, 1], 'r--')
        plt.scatter(y_test, y_pred)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted Security Risk')
        plt.savefig(FOLDER / 'riskpreds.png')
        plt.close()

        import seaborn as sns
        df = pd.DataFrame({'y_true': y_test.flatten(), 'y_pred': y_pred.flatten()})
        data = pd.DataFrame()
        for k in np.unique(df['y_true']):
            data[k] = pd.Series(df.loc[df['y_true'] == k]['y_pred'].values)
        data.columns = [round(col, 2) for col in data.columns]
        sns.boxplot(data=data, palette="Set1", showfliers=False)
        plt.savefig(FOLDER / 'boxplot.png')
        plt.close()
    elif risk:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        plot_cm(
            y_test,
            y_pred,
            FOLDER / 'cm.png',
            f'{datetime.now()}',
            p=0.5,
            risk=True
        )
    else:
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
    main(risk=True, regression=False)
