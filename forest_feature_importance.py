
from random import sample
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from preprocessing import columns

df = pandas.read_csv('dados/normalizado-preprocessado-publico.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

regr = RandomForestRegressor(**{'bootstrap': True, 'criterion': 'friedman_mse', 'max_samples': 0.4, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 100})
regr = regr.fit(X_train, y_train)

imp = pandas.Series(regr.feature_importances_, index=columns)
imp.sort_values(inplace=True)
print(imp[-15:].values)
print(imp[-15:].values.sum())
print(imp.values.sum())

fig, ax = plt.subplots(figsize=(12, 8))
imp[-15:].plot.bar(ax=ax)
print('Features consideradas: ', imp[imp != 0].count())
ax.set_title('MDI das 15 features mais importantes - Universidade pública')
ax.set_ylabel("Mean Decrease Impurity")
fig.tight_layout()
fig.savefig('forest_importance_publico.png')

#for i in sample(range(len(regr.estimators_)), k=5):
#    fig, ax = plt.subplots(figsize=(20, 20))
#    plot_tree(regr.estimators_[i], feature_names=columns, ax=ax)
#    fig.savefig(f'tree_{i}.png')

