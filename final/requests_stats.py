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
    print(pd.DataFrame({'action': y_a, 'riskscore': pd.Series(y_r.flatten())}).value_counts())


if __name__ == '__main__':
    riskscore_stats()
