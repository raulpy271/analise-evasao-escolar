
import numpy
import pandas
from sklearn.preprocessing import normalize

ignore = set(['PERC_DESVINCULADO'])
df = pandas.read_csv('processed-dados.csv', sep=';', encoding='latin-1')

for col, series in df.items():
    if col in ignore:
        continue
    df[col] = normalize([series])[0]

print(df.shape)
df.to_csv('normalized-dados.csv', sep=';', index=False, encoding='latin-1')
