import collections
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import toml
from cryptography.fernet import Fernet
from matplotlib.legend_handler import HandlerLineCollection
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


def load_config() -> dict:
    config = toml.load('config.toml')

    # Parse paths
    config['run_params']['logdir'] = Path(config['run_params']['logdir'])
    config['run_params']['data_src'] = Path(config['run_params']['data_src'])

    # Parse num_units
    if any('num_units' in x for x in config['hyperparameters']):
        val = config['hyperparameters']['hp_num_units']
        config['hyperparameters']['hp_num_units'] = hp.HParam(
            'num_units', hp.Discrete(val))
    else:
        config['hyperparameters']['hp_num_units'] = hp.HParam(
            'num_units', hp.Discrete([100]))

    # Parse batch_size
    if any('batch_size' in x for x in config['hyperparameters']):
        val = config['hyperparameters']['hp_batch_size']
        config['hyperparameters']['hp_batch_size'] = hp.HParam(
            'batch_size', hp.Discrete(val))
    else:
        config['hyperparameters']['hp_batch_size'] = hp.HParam(
            'batch_size', hp.Discrete([100]))

    # Parse dropout layer
    if any('dropout' in x for x in config['hyperparameters']):
        val = config['hyperparameters']['hp_dropout']
        assert isinstance(val, list)
        assert len(val) == 2
        assert isinstance(val[0], float)
        assert isinstance(val[1], float)
        config['hyperparameters']['hp_dropout'] = hp.HParam(
            'dropout', hp.RealInterval(val[0], val[1]))
    # else:
    # config['hyperparameters']['hp_dropout'] = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

    # Parse optimizer
    if any('optimizer' in x for x in config['hyperparameters']):
        val = config['hyperparameters']['hp_optimizer']
        config['hyperparameters']['hp_optimizer'] = hp.HParam(
            'optimizer', hp.Discrete(val))
    else:
        config['hyperparameters']['hp_optimizer'] = hp.HParam(
            'optimizer', hp.Discrete(['adam', 'sgd']))

    # List active hyperparameters
    config['hyperparameters']['hparams'] = [
        config['hyperparameters'][param] for param in config['hyperparameters']
        if 'hp_' in param
    ]
    return config


def flatten(l):
    for el in l:
        if isinstance(
                el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


# determine tensorflow distribution strategy
def dist_strategy(logger=None):
    # TPU detection
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        tpu = None

    # Select distribution strategy
    if tpu:
        tf.config.set_soft_device_placement(True)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        dist_strat = tf.distribute.experimental.TPUStrategy(tpu)
        if logger:
            logger.debug('Running on TPU %s',
                         tpu.cluster_spec().as_dict()['worker'])
    else:
        # GPU detection
        gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        if logger:
            logger.debug("Num GPUs Available: %s", gpus)
            logger.debug('Running on %s', 'GPU' if gpus > 0 else 'CPU')

        # Default strategy that works on CPU and single GPU
        dist_strat = tf.distribute.get_strategy()
    return dist_strat


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='matthews_correlation_coefficient', **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name,
                                                             **kwargs)
        self.true_pos = self.add_weight(name='true_pos', initializer='zeros')
        self.true_neg = self.add_weight(name='true_neg', initializer='zeros')
        self.false_pos = self.add_weight(name='false_pos', initializer='zeros')
        self.false_neg = self.add_weight(name='false_neg', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        prediction = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        label = tf.cast(y_true, tf.float32)

        true_pos = tf.math.count_nonzero(prediction * label)
        true_neg = tf.math.count_nonzero((prediction - 1) * (label - 1))
        false_pos = tf.math.count_nonzero(prediction * (label - 1))
        false_neg = tf.math.count_nonzero((prediction - 1) * label)

        true_pos = tf.cast(true_pos, tf.float32)
        true_neg = tf.cast(true_neg, tf.float32)
        false_pos = tf.cast(false_pos, tf.float32)
        false_neg = tf.cast(false_neg, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp_sample_weight = tf.broadcast_to(sample_weight, true_pos.shape)
            tn_sample_weight = tf.broadcast_to(sample_weight, true_neg.shape)
            fp_sample_weight = tf.broadcast_to(sample_weight, false_pos.shape)
            fn_sample_weight = tf.broadcast_to(sample_weight, false_neg.shape)

            true_pos = tf.multiply(true_pos, tp_sample_weight)
            true_neg = tf.multiply(true_neg, tn_sample_weight)
            false_pos = tf.multiply(false_pos, fp_sample_weight)
            false_neg = tf.multiply(false_neg, fn_sample_weight)
        self.true_pos.assign_add(true_pos, tf.float32)
        self.true_neg.assign_add(true_neg, tf.float32)
        self.false_pos.assign_add(false_pos, tf.float32)
        self.false_neg.assign_add(false_neg, tf.float32)

    def result(self):
        x = tf.cast((self.true_pos + self.false_pos) *
                    (self.true_pos + self.false_neg) *
                    (self.true_neg + self.false_pos) *
                    (self.true_neg + self.false_neg), tf.float32)
        mcc = tf.cast(
            (self.true_pos * self.true_neg) -
            (self.false_pos * self.false_neg), tf.float32) / tf.sqrt(x)
        mcc = tf.where(tf.math.is_nan(mcc), 0., mcc)
        return mcc


def plot_cm(y_test, y_pred, logdir_fig, paramset, p=0.5):
    sns.heatmap(confusion_matrix(y_test, y_pred),
                cmap="YlGnBu",
                annot=True,
                fmt="d")
    plt.title(f'{paramset}\n'
              f'Confusion matrix @{p:.2f}, '
              f'n={len(y_test)}, '
              f'Accuracy={round(accuracy_score(y_test, y_pred), 2)}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(logdir_fig)
    plt.close()


def plot_roc(labels, predictions, logdir_fig, paramset):
    def youdens_j_stat(fpr, tpr, thresholds):
        sensitivity = tpr
        specificity = 1. - fpr
        J = sensitivity + specificity - 1.
        opt_idx = np.argmax(J)
        opt_J = J[opt_idx]
        opt_threshold = thresholds[opt_idx]
        return opt_idx, opt_threshold, opt_J

    def make_segments(x, y, reverseColors=False):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        if reverseColors:
            points = np.flip(points, axis=0)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def colorline(x,
                  y,
                  z=None,
                  cmap=plt.get_cmap('copper'),
                  norm=plt.Normalize(0.0, 1.0),
                  linewidth=3,
                  alpha=1.0):

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(
                z,
                "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        lc = mcoll.LineCollection(segments,
                                  array=z,
                                  cmap=cmap,
                                  norm=norm,
                                  linewidth=linewidth,
                                  alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    class HandlerColorLineCollection(HandlerLineCollection):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            x = np.linspace(0, width, self.get_numpoints(legend) + 1)
            y = np.zeros(self.get_numpoints(legend) +
                         1) + height / 2. - ydescent
            segments = make_segments(x, y, reverseColors=True)
            lc = mcoll.LineCollection(segments,
                                      cmap=orig_handle.cmap,
                                      transform=trans)
            lc.set_array(x)
            lc.set_linewidth(orig_handle.get_linewidth())
            return [lc]

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    # thresholds[0] = 1.
    roc_auc = auc(fpr, tpr)
    opt_idx, opt_threshold, opt_J = youdens_j_stat(fpr, tpr, thresholds)

    lc = colorline(fpr, tpr, thresholds, cmap='YlGnBu', linewidth=2)
    cbar = plt.colorbar(lc)
    cbar.set_label('Discrimination threshold', rotation=270, labelpad=12)

    plt.plot([0, 1], [0, 1], 'r--', label='Random or constant class')
    plt.scatter(fpr[opt_idx],
                tpr[opt_idx],
                marker='o',
                color='black',
                label="Optimal threshold = %0.2f" % opt_threshold)
    plt.vlines(x=fpr[opt_idx],
               ymin=fpr[opt_idx],
               ymax=tpr[opt_idx],
               colors='black',
               linestyles='dashed',
               label="Youden's J statistic = %0.2f" % opt_J)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(lc)
    labels.append('ROC AUC = %0.2f' % roc_auc)
    order = [3, 1, 2, 0]
    plt.legend(handles=[handles[idx] for idx in order],
               labels=[labels[idx] for idx in order],
               loc='lower right',
               handler_map={lc: HandlerColorLineCollection(numpoints=5)})

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(f'{paramset}\nReceiver Operating Characteristic')
    plt.ylabel('True Positive Rate: TP/(TP+FN)')
    plt.xlabel('False Positive Rate: FP/(FP+TN)')
    plt.savefig(logdir_fig)
    plt.close()
    return opt_threshold


def plot_metrics(history: keras.callbacks.History,
                 optimizer: str,
                 loss: str,
                 figdir: str = None,
                 show: bool = False):
    # print(plt.style.available)
    plt.style.use('default')  # reset style
    plt.style.use('seaborn-ticks')
    fig, ax1 = plt.subplots()
    loss_color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=loss_color)
    # ax1.set_yscale('log')
    ax1.set_xlim(history.epoch[0] - 1, history.epoch[-1] + 1)
    ax1.set_ylim(
        0, 1.05 *
        max(history.history.get('loss') + history.history.get('val_loss')))
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('(val_)' + str([
        label for label in history.history.keys()
        if 'loss' not in label and 'val' not in label
    ]),
                   color='tab:green')
    ax2_min = 0
    for label in history.history.keys():
        if 'loss' not in label:
            ymin = min(history.history.get(label))
            if ymin < ax2_min:
                ax2_min = ymin
    ax2.set_ylim(1.05 * ax2_min, 1)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    acc_color = 'tab:green'
    auc_color = 'tab:blue'
    mcc_color = 'tab:purple'

    for label in history.history.keys():
        x = history.epoch
        alpha = 1.0
        if 'loss' in label:
            ax = ax1
            color = loss_color
            if 'val' not in label:
                # training loss is measured during each epoch while validation loss is measured
                # after each epoch, so on average training loss is measured half an epoch earlier
                x = [epoch - 0.5 for epoch in history.epoch]
                alpha = 0.4
        else:
            ax = ax2
            if 'accuracy' in label:
                color = acc_color
            elif 'MCC' in label:
                color = mcc_color
            else:
                color = auc_color
            if 'val' not in label:
                alpha = 0.4
        ax.plot(x,
                history.history.get(label),
                color=color,
                alpha=alpha,
                label=label)

    handles, labels = [], []
    for ax in fig.axes:
        handle, label = ax.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc='lower left')

    ax1.set_title(f"optimizer={optimizer}, loss={loss}")
    fig.tight_layout()
    if figdir is not None:
        fig.savefig(figdir)
    if show:
        plt.show()
    else:
        plt.close()


def send_notification(first, second, whatsapp=True):
    try:
        key = open("twilio.key", "rb").read()
        f = Fernet(key)
        with open("twilio.token", "rb") as file:
            account_sid_encrypted, auth_token_encrypted, from_encrypted, to_encrypted = file.readlines(
            )

        account_sid = f.decrypt(account_sid_encrypted).decode()
        auth_token = f.decrypt(auth_token_encrypted).decode()
        from_ = f.decrypt(from_encrypted).decode()
        to = f.decrypt(to_encrypted).decode()

        client = Client(account_sid, auth_token)

        if whatsapp:
            message = client.messages.create(
                from_=from_,
                body=f'Your {first} code is {second}',
                to=to,
            )
        else:
            message = client.messages.create(
                messaging_service_sid='MG97070ce0097d0b5b8a5279bd264c2e5c',
                body=f'Your {first} code is {second}',
                to='+31616302870')
    except FileNotFoundError as e:
        print(f"Failed to send notification: file {e.filename} not found")
    except TwilioRestException as e:
        print(f"Failed to send notification: {e.status}")
    return message


def plot_parallel_coordinates():
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
    #     X=pcdf,
    #     y=optcol,
    #     features=list(pcdf.columns),
    #     classes=['adam', 'sgd'])
    pass
