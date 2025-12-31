

import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import normalize

ignore = set(['TADA'])


for filename in ['preprocessado-indicadores', 'preprocessado-privado-indicadores', 'preprocessado-publico-indicadores']:
    df = pandas.read_csv(f'dados/{filename}.csv', sep=';', encoding='latin-1')

    for col, series in df.items():
        if col in ignore:
            continue
        df[col] = normalize([series])[0]

    print(df.shape)
    df.to_csv(f'dados/normalizado-{filename}.csv', sep=';', index=False, encoding='latin-1')
