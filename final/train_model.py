import csv
from datetime import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
from pathlib import Path
from time import perf_counter
from tensorflow import keras
from sklearn.model_selection import ParameterGrid, KFold

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

def create_model(input_shape, y_shape, paramset):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(input_shape, ), name='inputs'))
    if paramset['layers'] > 0:
        model.add(keras.layers.Dense(
                    paramset['neurons'],
                    input_shape=(input_shape, ),
                    activation=tf.nn.relu,
                    kernel_regularizer=paramset['regularizer'],
                    name='dense_1'))
        if paramset['layers'] > 1:
            for i in range(1, paramset['layers']):
                model.add(keras.layers.Dense(
                    paramset['neurons'],
                    input_shape=(paramset['neurons'], ),
                    activation=tf.nn.relu,
                    kernel_regularizer=paramset['regularizer'],
                    name=f'dense_{i+1}'))
        output_input_shape = paramset['neurons']
    else:
        output_input_shape = input_shape

    # Select output layer
    if paramset['model'] == 'multiclass':
        model.add(keras.layers.Dense(
                y_shape,
                input_shape=(output_input_shape, ),
                activation=keras.activations.softmax,
                name='output'))
    elif paramset['model'] == 'regression':
        model.add(keras.layers.Dense(
                y_shape,
                input_shape=(output_input_shape, ),
                activation=keras.activations.linear,
                name='output'))
    else:
        model.add(keras.layers.Dense(
                y_shape,
                input_shape=(output_input_shape, ),
                activation=tf.nn.sigmoid,
                name='output'))
    return model

def tune_hparams():
    hparams = {
        'model': [
            'binary',
            'multiclass',
            'regression',
        ],
        'optimizer': [
            'Adam',
            'RMSprop',
            # 'SGD',
        ],
        'layers': [2, 1, 0],
        'neurons': [100, 50, 3],
        'batch_size': [4096, 1024],
        'regularizer': [
            None,
            # keras.regularizers.l2(),
        ],
    }

    keras.utils.get_custom_objects().update(
        {'MCC': MatthewsCorrelationCoefficient})

    npzfile = np.load(FOLDER / 'Xys.npz')
    X = npzfile['X']
    y_b = npzfile['y_a']
    y_r = npzfile['y_r']
    y_m = pd.get_dummies(y_r.flatten()).values

    stats = []
    param_grid = ParameterGrid(hparams)

    fmt = "Progress: {:>3}% estimated {:>3}s remaining"
    num = len(param_grid)

    start = perf_counter()
    for session, paramset in enumerate(param_grid):
        stats.append(paramset)
        paramset_monitor_vals = []
        print(f'{datetime.now()} {session+1}/{len(param_grid)} starting {str(paramset)}')
        if paramset['model'] == 'regression':
            y = y_r
            loss = 'mean_squared_error'
            metrics = [keras.metrics.RootMeanSquaredError()]
            monitor = 'val_root_mean_squared_error'
            earlystopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=0,
                patience=50,
                verbose=1,
                mode="min",
                )
        elif paramset['model'] == 'multiclass':
            y = y_m
            loss = 'categorical_crossentropy'
            metrics = ['categorical_accuracy']
            monitor = 'val_categorical_accuracy'
            earlystopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=0,
                patience=50,
                verbose=1,
                mode="max",
                )
        else:
            y = y_b
            loss = 'binary_crossentropy'
            metrics = ['MCC']
            monitor = 'val_MCC'
            earlystopping = keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=0,
                patience=50,
                verbose=1,
                mode="max",
                )

        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=4).split(X, y)):
            # print(f'{datetime.now()} {session}-{fold+1} starting')
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            keras.backend.clear_session()

            model = create_model(
                input_shape=X_train.shape[1],
                y_shape=y_train.shape[1],
                paramset=paramset
            )

            model.compile(
                optimizer=paramset['optimizer'],
                loss=loss,
                metrics=metrics
            )

            history = model.fit(
                X_train, y_train,
                batch_size=paramset['batch_size'],
                epochs=1000,
                verbose=0,
                callbacks=[earlystopping],
                validation_data=(X_val, y_val)
            )
            if paramset['model'] == 'regression':
                monitor_val = min(history.history[monitor])
            else:
                monitor_val = max(history.history[monitor])
            paramset_monitor_vals.append(monitor_val)
            print(f'{datetime.now()} {session+1}-{fold+1} {round(monitor_val, 4)}')
        paramset_avg_monitor_val = np.mean(paramset_monitor_vals)
        stats[session]['monitor'] = monitor
        stats[session]['monitor_val'] = paramset_avg_monitor_val
        
        with open(FOLDER / 'hparams.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = []
            row.append(monitor)
            row.append(paramset_avg_monitor_val)
            row.append(paramset['model'])
            row.append(paramset['layers'])
            row.append(paramset['neurons'])
            row.append(paramset['batch_size'])
            if paramset['regularizer'] is None:
                row.append('None')
            else:
                row.append(paramset['regularizer']._keras_api_names[0])
            row.append(paramset['optimizer'])
            row.extend(paramset_monitor_vals)
            writer.writerow(row)

        stop = perf_counter()
        remaining = round((stop - start) * (num / (session+1) - 1))
        print(f'{datetime.now()}', fmt.format(100 * (session+1) // num, remaining), end='\n', flush=True)
        print(f'{datetime.now()} {session+1} {round(paramset_avg_monitor_val, 4)}')

    return pd.DataFrame(stats)


def main(model='binary'):
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
    y_b = npzfile['y_a']
    y_r = npzfile['y_r']
    y_m = pd.get_dummies(y_r.flatten()).values

    if model == 'regression':
        y = y_r
    elif model == 'multiclass':
        y = y_m
    else:
        y = y_b
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

    paramset = {
        'model': model,
        'layers': 0,
        'neurons': 3,
        'batch_size': 4096,
        'regularizer': None,
        'optimizer': 'Adam'
    }
    model = create_model(X_train.shape[1], y.shape[1], paramset)

    print(model.summary())

    if model == 'binary':
        loss = 'binary_crossentropy'
    elif model == 'multiclass':
        loss = 'categorical_crossentropy'
    else:
        loss = 'mean_squared_error'

    keras.utils.get_custom_objects().update(
        {'MCC': MatthewsCorrelationCoefficient})

    if model == 'multiclass':
        metrics = ['categorical_accuracy']
    elif model == 'regression':
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
        callbacks=[keras.callbacks.EarlyStopping(
                monitor='MCC',
                min_delta=0,
                patience=100,
                verbose=1,
                mode="max",
                )],
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
    if model == 'regression':
        plt.plot([0, 1], [0, 1], 'r--')
        plt.scatter(y_test, y_pred)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted Security Risk')
        plt.savefig(FOLDER / 'riskpreds.png')
        plt.close()

        df = pd.DataFrame({'y_true': y_test.flatten(), 'y_pred': y_pred.flatten()})
        data = pd.DataFrame()
        for k in np.unique(df['y_true']):
            data[k] = pd.Series(df.loc[df['y_true'] == k]['y_pred'].values)
        data.columns = [round(col, 2) for col in data.columns]
        sns.boxplot(data=data, palette="Set1", showfliers=False)
        plt.savefig(FOLDER / 'boxplot.png')
        plt.close()
    elif model == 'multiclass':
        plot_cm(
            np.argmax(y_test, axis=1),
            np.argmax(y_pred, axis=1),
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
    keras.backend.clear_session()


if __name__ == '__main__':
    # binary/multiclass/regression
    # main(model='binary')
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.backend.clear_session()

    # try:
    #     Path.unlink(FOLDER / 'hparams.csv')
    # except FileNotFoundError:
    #     pass
    # with open(FOLDER / 'hparams.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     headers = []
    #     headers.append('monitor')
    #     headers.append('monitor_val')
    #     headers.extend(list(hparams.keys()))
    #     writer.writerow(headers)
    df = tune_hparams()
    df.to_csv(FOLDER / 'hparams_complete.csv')
    print(df)
    keras.backend.clear_session()
