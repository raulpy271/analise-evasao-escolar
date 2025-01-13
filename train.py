import pandas
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

from preprocessing import columns

df = pandas.read_csv('normalized-dados.csv', sep=';', encoding='latin-1')
X, y = df[columns], df['PERC_DESVINCULADO']
metrics = ['mean_absolute_error', 'median_absolute_error', 'mean_squared_error', 'root_mean_squared_error']
scoring = ['neg_' + m for m in metrics]

models = {
    'linear_reg': LinearRegression(),
    'mlp': MLPRegressor(**{'activation': 'relu', 'hidden_layer_sizes': (200,), 'learning_rate': 'constant', 'max_iter': 100, 'solver': 'lbfgs'}),
    'svm': SVR(**{'coef0': 0, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'shrinking': False}),
    'd_tree': DecisionTreeRegressor(**{'criterion': 'poisson', 'min_samples_leaf': 3, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.1, 'splitter': 'random'}),
    'forest': RandomForestRegressor(**{'bootstrap': True, 'criterion': 'friedman_mse', 'max_samples': 0.4, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 100}),
}

columns_result = ['model', 'fold', 'train_time'] + metrics
data_result = {k: [] for k in columns_result}

for model, regr in models.items():
    folds = 5
    result = cross_validate(regr, X, y, cv=folds, scoring=scoring)
    for fold in range(folds):
        data_result['model'].append(model)
        data_result['fold'].append(fold + 1)
        data_result['train_time'].append(result['fit_time'][fold])
        for metric in metrics:
            data_result[metric].append(abs(result[f'test_neg_{metric}'][fold]))
    print('Modelo treinado', model)

df_result = pandas.DataFrame(data_result)

print(df_result)

df_result.to_csv('train_result.csv', sep=';', index=False)

