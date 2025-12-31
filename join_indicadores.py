
import pandas


indicadores_df = pandas.read_csv('dados/indicadores-geral.csv', sep=';')

for filename in ['preprocessado', 'preprocessado-privado', 'preprocessado-publico']:
    df = pandas.read_csv(f'dados/{filename}.csv', sep=';', encoding='latin-1')
    data = []
    for i in range(df.shape[0]):
        ind_curso = indicadores_df[indicadores_df["CO_CURSO_IND"] == df.iloc[i]["CO_CURSO"]]
        if not ind_curso.empty:
            data.append(
                df.iloc[i].to_list() + [ind_curso.iloc[0]["TADA"]]
            )
            #print("Encontrado: ", df.iloc[i]["CO_CURSO"], df.iloc[i]["TP_REDE"])
        #else:
        #    print("Não encontrado: ", df.iloc[i]["CO_CURSO"], df.iloc[i]["TP_REDE"])
    new_columns = list(df.columns) + ["TADA"]
    df = pandas.DataFrame(data, columns=new_columns)
    df = df.drop(["CO_CURSO"], axis=1)
    print(df.shape)
    df.to_csv(f'dados/{filename}-indicadores.csv', sep=';', index=False, encoding='latin-1')

