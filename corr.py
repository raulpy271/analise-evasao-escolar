import pandas

from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']

print(X.corrwith(y).dropna().sort_values())
