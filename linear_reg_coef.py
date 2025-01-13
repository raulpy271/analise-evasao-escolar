
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

regr = LinearRegression()
regr = regr.fit(X_train, y_train)
coefs = pandas.Series(regr.coef_, index=columns)
coefs.sort_values(inplace=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
coefs[:15].plot.bar(ax=ax1)
ax1.set_title('Os 15 menores coeficientes da Reg Linear')
coefs[-15:].plot.bar(ax=ax2)
ax2.set_title('Os 15 maiores coeficientes da Reg Linear')
fig.tight_layout()
fig.savefig('coefs.png')


