from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from generate_pips import FOLDER

df = pd.read_csv(FOLDER / 'requests.csv', index_col=0)

hours = []
daysofweek = []
for date in df['date']:
    daysofweek.append(datetime.strptime(date, '%Y-%m-%d').weekday())
for time in df['time']:
    hours.append(datetime.strptime(time[-8:], '%H:%M:%S').hour)
c = Counter(daysofweek)
freqs = [i[1] for i in sorted(c.items())]
print(freqs)
plt.bar(range(len(freqs)), freqs)
plt.show()
