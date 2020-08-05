import collections
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
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
            logger.debug('Running on TPU %s', tpu.cluster_spec().as_dict()['worker'])
    else:
        # GPU detection
        gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        if logger:
            logger.debug("Num GPUs Available: %s", gpus)
            logger.debug('Running on %s', 'GPU' if gpus > 0 else 'CPU')

        # Default strategy that works on CPU and single GPU
        dist_strat = tf.distribute.get_strategy()
    return dist_strat

def plot_history(history: keras.callbacks.History, optimizer: str, loss: str, figdir: str = None, show: bool = False):  # pylint: disable=line-too-long
    # print(plt.style.available)
    plt.style.use('default') # reset style
    plt.style.use('seaborn-ticks')
    fig, ax1 = plt.subplots()
    loss_color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=loss_color)
    ax1.set_xlim(history.epoch[0]-1, history.epoch[-1]+1)
    ax1.set_ylim(0, 1.05 * max(history.history.get('loss') + history.history.get('val_loss')))
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    acc_color = 'tab:green'
    ax2.set_ylabel('accuracy', color=acc_color)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor=acc_color)

    for label in history.history.keys():
        x = history.epoch
        alpha = 1.0
        if 'loss' in label:
            ax = ax1
            color = loss_color
            if 'val' not in label:
                # training loss is measured during each epoch while validation loss is measured
                # after each epoch, so on average training loss is measured half an epoch earlier
                x = [epoch-0.5 for epoch in history.epoch]
                alpha = 0.4
        else:
            ax = ax2
            color = acc_color
            if 'val' not in label:
                alpha = 0.4
        ax.plot(x, history.history.get(label), color=color, alpha=alpha, label=label)

    lines, labels = [], []
    for ax in fig.axes:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    ax1.legend(lines, labels, loc='lower left')

    ax1.set_title(f"optimizer={optimizer}, loss={loss}")
    fig.tight_layout()
    if figdir is not None:
        fig.savefig(figdir)
    if show:
        plt.show()
    else:
        plt.close()