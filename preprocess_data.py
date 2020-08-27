from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def preprocess_data(data_src=Path("data.csv"), seed=1, plot=False, logger=None):
    df = pd.read_csv(data_src)

    del df['ID']
    del df['date_time']
    del df['lat']
    del df['lng']


    for col in df.columns:
        if df[col].dtype == object:
            one_hot = pd.get_dummies(df[col])
            del df[col]
            df = df.join(one_hot)

    if plot:
        df['clearance'].value_counts().plot(kind='pie')
        plt.show()
        df['confi_lvl'].value_counts().plot(kind='pie')
        plt.show()
        plt.matshow(df.filter([
            'Aaliyah Otto-Post',
            'clearance',
            'confi_lvl',
            'access_granted'
            ]).corr(), cmap=plt.get_cmap('afmhot_r'), interpolation='none')
        plt.gca().xaxis.tick_bottom()
        plt.gca().set_title('correlation_matrix')
        plt.show()

    y = df.pop('access_granted')

    if plot:
        y.value_counts().plot(kind='pie')
        plt.show()

    del df['clearance']
    del df['confi_lvl']

    # TPUs cant handle uint8, GPU cant handle uint32
    df = df.astype(np.int32)
    y = y.astype(np.int32)
    if logger:
        logger.debug(df.dtypes)

    X = df.values
    y = y.values
    X, y = shuffle(X, y, random_state=seed)

    return X, y
