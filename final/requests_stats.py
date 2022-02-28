from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_pips import FOLDER

rdf = pd.read_csv(FOLDER / 'requests.csv', index_col=0)
ldf = pd.read_csv(FOLDER / 'labels.csv', index_col=0)


def show_date_freqs():
    daysofweek = []
    for date in rdf['date']:
        daysofweek.append(datetime.strptime(date, '%Y-%m-%d').weekday())
    c = Counter(daysofweek)
    freqs = [i[1] for i in sorted(c.items())]
    print(freqs)
    plt.bar(range(len(freqs)), freqs)
    plt.show()

def show_time_freqs():
    hours = []
    for time in rdf['time']:
        hours.append(datetime.strptime(time[-8:], '%H:%M:%S').hour)
    c = Counter(hours)
    freqs = [i[1] for i in sorted(c.items())]
    print(freqs)
    plt.bar(range(len(freqs)), freqs)
    plt.show()

def riskscore_stats():
    # print(ldf.value_counts())
    npzfile = np.load(FOLDER / 'Xys.npz')
    X = npzfile['X']
    y_a = npzfile['y_a']
    y_r = npzfile['y_r']
    print(pd.DataFrame({'action': y_a.flatten(), 'riskscore': pd.Series(y_r.flatten())}).value_counts())

def hparams_stats():
    df = pd.read_csv(FOLDER / FOLDER / 'hparams_final.csv').fillna('None')
    # print(df.groupby('model')[['monitor', 'monitor_val']].describe()[[('monitor_val', 'mean'), ('monitor_val',   'std')]])
    model = 'binary'
    modeldf = df.loc[df['model'] == model]
    print(modeldf)
    if model == 'regression':
        modelbest = modeldf.iloc[np.argmin(modeldf['monitor_val'])].copy()
    else:
        modelbest = modeldf.iloc[np.argmax(modeldf['monitor_val'])].copy()
    # modelbest.drop(['monitor', 'monitor_val', 'f1', 'f2', 'f3', 'f4'], inplace=True)
    modelbest = modelbest.to_dict()
    print(modelbest)

    # print(modeldf.groupby('layers')[['monitor_val']].describe())

if __name__ == '__main__':
    riskscore_stats()
