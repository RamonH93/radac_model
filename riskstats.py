import pandas as pd

df = pd.read_csv('riskdataset.csv')

print(df.groupby('confi_lvl').describe())
