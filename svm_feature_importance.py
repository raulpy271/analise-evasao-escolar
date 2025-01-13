
from random import sample
import pandas
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

regr = SVR(**{'coef0': 0, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'shrinking': False})
regr = regr.fit(X_train, y_train)

r = permutation_importance(regr, X_test, y_test, n_repeats=30, random_state=0)
imp = pandas.Series(r.importances_mean, index=columns)
imp.sort_values(inplace=True)
print('Features consideradas', imp[imp != 0].count())

fig, ax = plt.subplots(figsize=(12, 8))
imp[-30:].plot.bar(ax=ax)
ax.set_title('Permutation Importance das 30 features mais importantes - Modelo SVM')
ax.set_ylabel("Permutation Importance")
fig.tight_layout()
fig.savefig('svm_importance.png')


