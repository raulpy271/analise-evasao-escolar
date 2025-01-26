
import pandas
from sklearn.preprocessing import normalize

ignore = set(['PERC_DESVINCULADO'])

for filename in ['preprocessado', 'preprocessado-privado', 'preprocessado-publico']:
    df = pandas.read_csv(f'dados/{filename}.csv', sep=';', encoding='latin-1')

    for col, series in df.items():
        if col in ignore:
            continue
        df[col] = normalize([series])[0]

    print(df.shape)
    df.to_csv(f'dados/normalizado-{filename}.csv', sep=';', index=False, encoding='latin-1')
