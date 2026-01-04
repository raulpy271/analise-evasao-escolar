
from random import sample
import pandas
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pandas.read_csv('dados/normalizado-preprocessado-publico-indicadores.csv', sep=';', encoding='latin-1')
y, X = df['TADA'], df.drop('TADA', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

regr = RandomForestRegressor(**{'bootstrap': True, 'criterion': 'squared_error', 'max_samples': 0.4, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.1, 'n_estimators': 100})

regr = regr.fit(X_train, y_train)

shap.plots._beeswarm.labels["FEATURE_VALUE_LOW"] = "Baixo"
shap.plots._beeswarm.labels["FEATURE_VALUE_HIGH"] = "Alto"
shap.plots._beeswarm.labels["VALUE"] = "Valor SHAP (Impacto no resultado do modelo)"


explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(
    shap_values,
    X_test,
    max_display=15,
    show=False,
    color_bar_label="Valor da variável de entrada",
)

#plt.title("Valor SHAP das 15 variáveis mais importantes - Universidade privada")

plt.savefig('images/shap_publico.png')
