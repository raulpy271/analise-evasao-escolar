
import pandas

df_file = '/home/raul/Downloads/microdados_censo_da_educacao_superior_2023/dados/MICRODADOS_CADASTRO_CURSOS_2023.CSV'
columns_all = [
    'TP_GRAU_ACADEMICO',
    'TP_REDE',
    'CO_CINE_AREA_GERAL',
    'IN_GRATUITO',
    'QT_CURSO',
    'QT_VG_TOTAL',
    'QT_VG_TOTAL_DIURNO',
    'QT_VG_TOTAL_NOTURNO',
    'QT_VG_TOTAL_EAD',
    'QT_VG_NOVA',
    'QT_VG_PROC_SELETIVO',
    'QT_VG_REMANESC',
    'QT_VG_PROG_ESPECIAL',
    'QT_INSCRITO_TOTAL',
    'QT_INSCRITO_TOTAL_DIURNO',
    'QT_INSCRITO_TOTAL_NOTURNO',
    'QT_INSCRITO_TOTAL_EAD',
    'QT_INSC_VG_NOVA',
    'QT_INSC_PROC_SELETIVO',
    'QT_INSC_VG_REMANESC',
    'QT_INSC_VG_PROG_ESPECIAL',
    'QT_ING',
    'QT_ING_FEM',
    'QT_ING_MASC',
    'QT_ING_DIURNO',
    'QT_ING_NOTURNO',
    'QT_ING_VG_NOVA',
    'QT_ING_VESTIBULAR',
    'QT_ING_ENEM',
    'QT_ING_AVALIACAO_SERIADA',
    'QT_ING_SELECAO_SIMPLIFICA',
    'QT_ING_EGR',
    'QT_ING_OUTRO_TIPO_SELECAO',
    'QT_ING_PROC_SELETIVO',
    'QT_ING_VG_REMANESC',
    'QT_ING_VG_PROG_ESPECIAL',
    'QT_ING_OUTRA_FORMA',
    'QT_ING_0_17',
    'QT_ING_18_24',
    'QT_ING_25_29',
    'QT_ING_30_34',
    'QT_ING_35_39',
    'QT_ING_40_49',
    'QT_ING_50_59',
    'QT_ING_60_MAIS',
    'QT_ING_BRANCA',
    'QT_ING_PRETA',
    'QT_ING_PARDA',
    'QT_ING_AMARELA',
    'QT_ING_INDIGENA',
    'QT_ING_CORND',
    'QT_MAT',
    'QT_MAT_FEM',
    'QT_MAT_MASC',
    'QT_MAT_DIURNO',
    'QT_MAT_NOTURNO',
    'QT_MAT_0_17',
    'QT_MAT_18_24',
    'QT_MAT_25_29',
    'QT_MAT_30_34',
    'QT_MAT_35_39',
    'QT_MAT_40_49',
    'QT_MAT_50_59',
    'QT_MAT_60_MAIS',
    'QT_MAT_BRANCA',
    'QT_MAT_PRETA',
    'QT_MAT_PARDA',
    'QT_MAT_AMARELA',
    'QT_MAT_INDIGENA',
    'QT_MAT_CORND',
    'QT_CONC',
    'QT_CONC_FEM',
    'QT_CONC_MASC',
    'QT_CONC_DIURNO',
    'QT_CONC_NOTURNO',
    'QT_CONC_0_17',
    'QT_CONC_18_24',
    'QT_CONC_25_29',
    'QT_CONC_30_34',
    'QT_CONC_35_39',
    'QT_CONC_40_49',
    'QT_CONC_50_59',
    'QT_CONC_60_MAIS',
    'QT_CONC_BRANCA',
    'QT_CONC_PRETA',
    'QT_CONC_PARDA',
    'QT_CONC_AMARELA',
    'QT_CONC_INDIGENA',
    'QT_CONC_CORND',
    'QT_ING_NACBRAS',
    'QT_ING_NACESTRANG',
    'QT_MAT_NACBRAS',
    'QT_MAT_NACESTRANG',
    'QT_CONC_NACBRAS',
    'QT_CONC_NACESTRANG',
    'QT_ALUNO_DEFICIENTE',
    'QT_ING_DEFICIENTE',
    'QT_MAT_DEFICIENTE',
    'QT_CONC_DEFICIENTE',
    'QT_ING_FINANC',
    'QT_ING_FINANC_REEMB',
    'QT_ING_FIES',
    'QT_ING_RPFIES',
    'QT_ING_FINANC_REEMB_OUTROS',
    'QT_ING_FINANC_NREEMB',
    'QT_ING_PROUNII',
    'QT_ING_PROUNIP',
    'QT_ING_NRPFIES',
    'QT_ING_FINANC_NREEMB_OUTROS',
    'QT_MAT_FINANC',
    'QT_MAT_FINANC_REEMB',
    'QT_MAT_FIES',
    'QT_MAT_RPFIES',
    'QT_MAT_FINANC_REEMB_OUTROS',
    'QT_MAT_FINANC_NREEMB',
    'QT_MAT_PROUNII',
    'QT_MAT_PROUNIP',
    'QT_MAT_NRPFIES',
    'QT_MAT_FINANC_NREEMB_OUTROS',
    'QT_CONC_FINANC',
    'QT_CONC_FINANC_REEMB',
    'QT_CONC_FIES',
    'QT_CONC_RPFIES',
    'QT_CONC_FINANC_REEMB_OUTROS',
    'QT_CONC_FINANC_NREEMB',
    'QT_CONC_PROUNII',
    'QT_CONC_PROUNIP',
    'QT_CONC_NRPFIES',
    'QT_CONC_FINANC_NREEMB_OUTROS',
    'QT_ING_RESERVA_VAGA',
    'QT_ING_RVREDEPUBLICA',
    'QT_ING_RVETNICO',
    'QT_ING_RVPDEF',
    'QT_ING_RVSOCIAL_RF',
    'QT_ING_RVOUTROS',
    'QT_MAT_RESERVA_VAGA',
    'QT_MAT_RVREDEPUBLICA',
    'QT_MAT_RVETNICO',
    'QT_MAT_RVPDEF',
    'QT_MAT_RVSOCIAL_RF',
    'QT_MAT_RVOUTROS',
    'QT_CONC_RESERVA_VAGA',
    'QT_CONC_RVREDEPUBLICA',
    'QT_CONC_RVETNICO',
    'QT_CONC_RVPDEF',
    'QT_CONC_RVSOCIAL_RF',
    'QT_CONC_RVOUTROS',
    'QT_SIT_TRANCADA',
    'QT_SIT_DESVINCULADO',
    'QT_SIT_TRANSFERIDO',
    'QT_SIT_FALECIDO',
    'QT_ING_PROCESCPUBLICA',
    'QT_ING_PROCESCPRIVADA',
    'QT_ING_PROCNAOINFORMADA',
    'QT_MAT_PROCESCPUBLICA',
    'QT_MAT_PROCESCPRIVADA',
    'QT_MAT_PROCNAOINFORMADA',
    'QT_CONC_PROCESCPUBLICA',
    'QT_CONC_PROCESCPRIVADA',
    'QT_CONC_PROCNAOINFORMADA',
    'QT_PARFOR',
    'QT_ING_PARFOR',
    'QT_MAT_PARFOR',
    'QT_CONC_PARFOR',
    'QT_APOIO_SOCIAL',
    'QT_ING_APOIO_SOCIAL',
    'QT_MAT_APOIO_SOCIAL',
    'QT_CONC_APOIO_SOCIAL',
    'QT_ATIV_EXTRACURRICULAR',
    'QT_ING_ATIV_EXTRACURRICULAR',
    'QT_MAT_ATIV_EXTRACURRICULAR',
    'QT_CONC_ATIV_EXTRACURRICULAR',
    'QT_MOB_ACADEMICA',
    'QT_ING_MOB_ACADEMICA',
    'QT_MAT_MOB_ACADEMICA',
    'QT_CONC_MOB_ACADEMICA'
]
columns = columns_all.copy()
columns.remove('QT_SIT_DESVINCULADO')
columns.remove('CO_CINE_AREA_GERAL')
columns.remove('TP_REDE')
columns_to_remove_zeros = ['TP_GRAU_ACADEMICO', 'QT_VG_TOTAL', 'QT_MAT', 'QT_SIT_DESVINCULADO']

if __name__ == '__main__':
    df = pandas.read_csv(df_file, sep=';', usecols=columns_all, encoding='latin-1')

    print(df.shape)
    for col in columns_to_remove_zeros:
        df = df[df[col] > 0]
        print(df.shape)

    df['PERC_DESVINCULADO'] = df['QT_SIT_DESVINCULADO'] / df['QT_MAT']
    df = df[df['PERC_DESVINCULADO'] <= 1]
    print(df.shape)

    # Filtra áreas da computação
    df = df[df['CO_CINE_AREA_GERAL'].isin([6])]
    print(df.shape)

    df_publico = df[df['TP_REDE'] == 1]
    print("Publico", df_publico.shape)
    df_privado = df[df['TP_REDE'] == 2]
    print("Privado", df_privado.shape)

    df.to_csv('dados/preprocessado.csv', sep=';', index=False, columns=columns + ['PERC_DESVINCULADO'], encoding='latin-1')
    df_publico.to_csv('dados/preprocessado-publico.csv', sep=';', index=False, columns=columns + ['PERC_DESVINCULADO'], encoding='latin-1')
    df_privado.to_csv('dados/preprocessado-privado.csv', sep=';', index=False, columns=columns + ['PERC_DESVINCULADO'], encoding='latin-1')

