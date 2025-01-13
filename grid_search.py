import pandas
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']
name = 'forest'
regrs = {
    'svm': (SVR(), {
        'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'degree': (2, 3, 4),
        'gamma': ('scale', 'auto'),
        'coef0': (0, 1, 10),
        'shrinking': (True, False),
    }),
    'd_tree': (DecisionTreeRegressor(), {
        'criterion': ('squared_error', 'friedman_mse', 'poisson'),
        'splitter': ('best', 'random'),
        'min_samples_leaf': tuple(range(1, 10, 2)),
        'min_samples_split': tuple(range(2, 20, 4)),
        'min_weight_fraction_leaf': (0.0, 0.1, 0.3, 0.5)
    }),
    'mlp': (MLPRegressor(), {
        'hidden_layer_sizes': ((200,), (300,), (400,)),
        'activation': ('logistic', 'tanh', 'relu'),
        'solver': ('lbfgs', 'sgd', 'adam'),
        'learning_rate': ('constant', 'invscaling', 'adaptive'),
        'max_iter': (100, 200, 300)
    }),
    'forest': (RandomForestRegressor(), {
        'n_estimators': (100, 200, 300),
        'bootstrap': (False, True),
        'max_samples': (0.4, 0.8, None),
        'criterion': ('squared_error', 'friedman_mse', 'poisson'),
        'min_samples_leaf': (1, 6, 8),
        'min_weight_fraction_leaf': (0.1, 0.3, 0.5)
    })
}
regr, params = regrs[name]
grid_regr = GridSearchCV(regr, params, scoring='neg_root_mean_squared_error', cv=5)
grid_regr.fit(X, y)

df = pandas.DataFrame(grid_regr.cv_results_)
df = df.sort_values(by='rank_test_score')

print(df)

df.to_csv(f'grid-search-{name}.csv', sep=';', index=False)

