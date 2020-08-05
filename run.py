import os
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import config as cf
import utils


def train_test_model(X_train, y_train, X_test, y_test, hparams, callbacks, dist_strat, show):
    model = keras.models.Sequential([
        keras.Input(shape=(X_train.shape[1],), batch_size=hparams[cf.HP_BATCH_SIZE], name='inputs'),
        # keras.Input(shape=list(dataset.take(1).as_numpy_iterator())[0][0].shape)) #tfdataset
        keras.layers.Dense(
            hparams[cf.HP_NUM_UNITS],
            input_shape=(X_train.shape[1],),
            activation=tf.nn.relu,
            name='dense'),
        # kernel_initializer=keras.initializers.Ones(),
        # bias_initializer=keras.initializers.Zeros(),
        keras.layers.Dense(
            1,
            input_shape=(hparams[cf.HP_NUM_UNITS],),
            activation=tf.nn.sigmoid,
            name='output'),
    ])
    with dist_strat.scope():
        model.compile(
            optimizer=hparams[cf.HP_OPTIMIZER],
            loss='binary_crossentropy',
            metrics=[cf.METRIC_ACCURACY],
        )
    figdir = 'logs/figs'
    figname = f'model_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    os.makedirs(figdir, exist_ok=True)
    figsrc = os.path.join(figdir, figname)
    keras.utils.plot_model(model, figsrc, show_shapes=True, rankdir='LR')
    if show:
        plt.axis('off')
        plt.imshow(mpimg.imread(figsrc))
        plt.show()

    history = model.fit(
        X_train, y_train,
        batch_size=hparams[cf.HP_BATCH_SIZE],
        epochs=50,
        verbose=0,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        shuffle=False)
    figdir = f'logs/figs/fig_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
    utils.plot_history(history, hparams[cf.HP_OPTIMIZER], 'binary_crossentropy', figdir, show=show)
    accuracy = max(history.history['val_accuracy'])
    return accuracy

def run(run_dir, X_train, y_train, X_test, y_test, paramset, callbacks, dist_strat, show=False):
    with tf.summary.create_file_writer(run_dir).as_default():
        hparams = {h: paramset[h.name] for h in cf.HPARAMS}
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    hparams,
                                    callbacks,
                                    dist_strat,
                                    show)
        tf.summary.scalar(cf.METRIC_ACCURACY, accuracy, step=1)

        # # parallel coordinates test
        # pcdf = pd.DataFrame(paramset, index=[0])
        # pcdf['best_val_accuracy'] = accuracy
        # print(pcdf)
        # pd.plotting.parallel_coordinates(pcdf, 'optimizer')
        # plt.show()
        # optcol = pcdf.pop('optimizer')
        # # bscol = pcdf.pop('batch_size')
        # from yellowbrick.features import Rank2D
        # pcdf = pcdf.to_numpy()
        # visualizer = Rank2D(algorithm="pearson")
        # visualizer.fit_transform(pcdf)
        # visualizer.draw()
        # yellowbrick.features.parallel_coordinates(pcdf, bscol)
        # opt = 1 if optcol[0] == 'adam' else 0
        # visualizer = yellowbrick.features.parallel_coordinates(
            # X=pcdf,
            # y=optcol,
            # features=list(pcdf.columns),
            # classes=['adam', 'sgd'])
