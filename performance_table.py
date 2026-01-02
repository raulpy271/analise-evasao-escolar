
import pandas

metrics = ['mean_absolute_error', 'median_absolute_error', 'mean_squared_error', 'root_mean_squared_error']
df_geral = pandas.read_csv('resultados/treino-geral.csv', sep=';')
df_publico = pandas.read_csv('resultados/treino-publico.csv', sep=';')
df_privado = pandas.read_csv('resultados/treino-privado.csv', sep=';')

col = "Modelo", "Tempo (s)", "MAE", "MSE", "RMSE"
models = {
    "d_tree": "DT",
    "forest": "RF",
    "mlp": "MLP",
    "svm": "SVM",
}
data = []

for model, name in models.items():
    df = df_privado[df_privado["model"] == model]
    time = df["train_time"]
    mae = df["mean_absolute_error"]
    mse = df["mean_squared_error"]
    rmse = df["root_mean_squared_error"]
    data.append([
        name,
        "{:.2f} / {:.3f}".format(time.mean(), time.std()),
        "{:.3f} / {:.3f}".format(mae.mean(), mae.std()),
        "{:.3f} / {:.3f}".format(mse.mean(), mse.std()),
        "{:.3f} / {:.3f}".format(rmse.mean(), rmse.std()),
    ])

df = pandas.DataFrame(data, columns=col)

df.to_latex("tables/performance_table_privado.tex", index=False, decimal=",")

