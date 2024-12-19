import pandas
import matplotlib.pyplot as plt

from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')

def label_value(v):
    if v < 0.10:
        return 'até 10%'
    elif v < 0.2:
        return '10% - 20%'
    elif v < 0.3:
        return '20% - 30%'
    elif v < 0.4:
        return '30% - 40%'
    else: return 'mais que 40%'

df['LABEL_DESVINCULADO'] = df['PERC_DESVINCULADO'].map(label_value)

print(df)

columns = list(df.columns)
columns.remove('PERC_DESVINCULADO')
df.to_csv('normalized-dados-label.csv', sep=';', columns=columns, index=False, encoding='latin-1')

fig, ax = plt.subplots(figsize=(6, 6))
ax.set(title='Distribuição do percentual de desvinculados')
ax.hist(df['LABEL_DESVINCULADO'])
fig.savefig('histogram.png')
