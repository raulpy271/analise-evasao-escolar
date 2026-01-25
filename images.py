import pandas
import matplotlib.pyplot as plt

#metrics = ['mean_absolute_error', 'median_absolute_error', 'mean_squared_error', 'root_mean_squared_error']
metrics = ['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']
metrics_out = ['MAE', 'MSE', 'RMSE']
df = pandas.read_csv('resultados/treino-publico.csv', sep=';')

models = {
    "d_tree": "DT",
    "forest": "RF",
    "mlp": "MLP",
    "svm": "SVM",
}

def create_boxplot_data(df, metric):
    data = {}
    for model, group in df.groupby('model'):
        if model in models:
            data[models[model]] = list(group[metric])
    return pandas.DataFrame(data)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    ax = axs[i]
    metric = metrics[i]
    data = create_boxplot_data(df, metric)
    ax.set(title=metrics_out[i])
    ax.boxplot(data, tick_labels=data.columns, showmeans=True, meanline=True, meanprops={"color": "red"}, medianprops={"color": "green"})

plt.subplots_adjust(left=0.05, right=0.95)
fig.savefig('images/boxplot_metricas_publico.png')

#fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#for i in range(2):
#    ax = axs[i]
#    metric = metrics[i + 2]
#    data = create_boxplot_data(df, metric)
#    ax.set(title='Boxplot para métrica: ' + metric)
#    ax.boxplot(data, tick_labels=data.columns, showmeans=True, meanline=True, meanprops={"color": "red"}, medianprops={"color": "green"})
#fig.savefig('metricas_02.png')
#
#fig, ax = plt.subplots(figsize=(10, 4))
#data = create_boxplot_data(df, 'train_time')
#ax.set(title='Boxplot para tempo de treinamento', xlim=(0, 140), xlabel='tempo (s)')
#ax.boxplot(data, tick_labels=data.columns, vert=False, showmeans=True, meanline=True, meanprops={"color": "red"}, medianprops={"color": "green"})
#
#fig.savefig('tempo-execucao.png')
