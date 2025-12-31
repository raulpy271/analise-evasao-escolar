
import pandas

filepath = "~/Downloads/indicadores_fluxo_es_2019-2023/indicadores_trajetoria_educacao_superior_2019_2023.xlsx"

cols_old = ["Unnamed: 4", "Unnamed: 13", "Unnamed: 16", "Unnamed: 30"]
cols_new = ["CO_CURSO_IND", "CO_CINE_AREA_GERAL_IND", "NU_ANO_REFERENCIA", "TADA"]
data_start = 8
ano_referencia = 2023
area_geral = "06"

df = pandas.read_excel(filepath)
df = df[data_start:-1]
df = df.drop(filter(lambda col: col not in cols_old, df.columns), axis=1)
df.columns = cols_new
df = df[df["NU_ANO_REFERENCIA"] == ano_referencia]
df = df[df["CO_CINE_AREA_GERAL_IND"] == area_geral]
df["TADA"] = df["TADA"].map(lambda t: t / 100)

print(df)

df.to_csv('dados/indicadores-geral.csv', sep=';', columns=["CO_CURSO_IND", "TADA"], index=False)

