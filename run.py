from datetime import datetime
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import utils


def train_test_model(run_name, X_train, y_train, X_test, y_test, paramset, callbacks, dist_strat, show=False): # pylint: disable=line-too-long
    config = utils.load_config()

    model = keras.models.Sequential([
        keras.Input(shape=(X_train.shape[1],), batch_size=paramset['batch_size'], name='inputs'),
        # keras.Input(shape=list(dataset.take(1).as_numpy_iterator())[0][0].shape)) #tfdataset
        keras.layers.Dense(
            paramset['num_units'],
            input_shape=(X_train.shape[1],),
            activation=tf.nn.relu,
            name='dense'),
        keras.layers.Dense(
            paramset['num_units']*2,
            input_shape=(X_train.shape[1],),
            activation=tf.nn.relu,
            name='dense_1'),
        keras.layers.Dense(
            paramset['num_units'],
            input_shape=(X_train.shape[1],),
            activation=tf.nn.relu,
            name='dense_2'),
        # kernel_initializer=keras.initializers.Ones(),
        # bias_initializer=keras.initializers.Zeros(),
        keras.layers.Dense(
            1,
            input_shape=(paramset['num_units'],),
            activation=tf.nn.sigmoid,
            name='output'),
    ])

    with dist_strat.scope():
        model.compile(
            optimizer=paramset['optimizer'],
            loss='binary_crossentropy',
            metrics=[config['hyperparameters']['metric_accuracy']],
        )

    Path.mkdir(config['model_loc'], parents=True, exist_ok=True)
    model_loc = config['model_loc'] / f'{run_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}-model.png'  # pylint: disable=line-too-long
    keras.utils.plot_model(model, model_loc, show_shapes=True, rankdir='LR')
    if show:
        plt.axis('off')
        plt.imshow(mpimg.imread(model_loc))
        plt.show()

    history = model.fit(
        X_train, y_train,
        batch_size=paramset['batch_size'],
        epochs=25,
        verbose=0,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        shuffle=False)

    Path.mkdir(config['fig_loc'], parents=True, exist_ok=True)
    fig_loc = config['fig_loc'] / f'{run_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}-fig.png'
    utils.plot_history(history, paramset['optimizer'], 'binary_crossentropy', fig_loc, show=show)
    accuracy = max(history.history['val_accuracy'])

    # explainer = LimeTabularExplainer(
    #             training_data=X_train,
    #             feature_names=[
    #                 'RESOURCE',
    #                 'MGR_ID',
    #                 'ROLE_ROLLUP_1',
    #                 'ROLE_ROLLUP_2',
    #                 'ROLE_DEPTNAME',
    #                 'ROLE_TITLE',
    #                 'ROLE_FAMILY_DESC',
    #                 'ROLE_FAMILY',
    #                 'ROLE_CODE'
    #                 ],
    #             class_names=['PERMIT', 'DENY'],
    #             discretize_continuous=True
    # )
    # explainer = LimeTabularExplainer(
    #             training_data=X_train,
    #             feature_names=[f'f_{x}' for x in range(len(X_train[0]))],
    #             class_names=['access_granted', 'not granted'],
    #             discretize_continuous=True
    # )
    # exp = explainer.explain_instance(X_test[1], model.predict, top_labels=1)
    # exp.save_to_file('explanation.html')


    return accuracy
