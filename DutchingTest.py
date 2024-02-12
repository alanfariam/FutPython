import pandas as pd

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

dia = '2024-01-14'

jogos_do_dia = pd.read_csv(f"https://github.com/futpythontrader/YouTube/raw/main/Base_de_Dados/Jogos_do_Dia_Betfair/Jogos_do_Dia_Betfair_Back_Lay_{dia}.csv")

jogos_do_dia.columns.to_list()

print(jogos_do_dia.columns.to_list())

jogos_do_dia = jogos_do_dia[['League','Date','Time','Home','Away',
'Odd_CS_0x0_Back','Odd_CS_0x1_Back','Odd_CS_0x2_Back','Odd_CS_0x3_Back',
'Odd_CS_1x0_Back','Odd_CS_1x1_Back','Odd_CS_1x2_Back','Odd_CS_1x3_Back',
'Odd_CS_2x0_Back','Odd_CS_2x1_Back','Odd_CS_2x2_Back','Odd_CS_2x3_Back',
'Odd_CS_3x0_Back','Odd_CS_3x1_Back','Odd_CS_3x2_Back','Odd_CS_3x3_Back',
'Odd_CS_Goleada_H_Back','Odd_CS_Goleada_A_Back','Odd_CS_Goleada_D_Back']]
jogos_do_dia.columns = ['League','Date','Time','Home','Away',
'Odd_0x0','Odd_0x1','Odd_0x2','Odd_0x3',
'Odd_1x0','Odd_1x1','Odd_1x2','Odd_1x3',
'Odd_2x0','Odd_2x1','Odd_2x2','Odd_2x3',
'Odd_3x0','Odd_3x1','Odd_3x2','Odd_3x3',
'Odd_Goleada_H','Odd_CS_Goleada_A','Odd_CS_Goleada_D']
jogos_do_dia = drop_reset_index(jogos_do_dia)
jogos_do_dia

flt = jogos_do_dia.Home == 'Real Madrid'
df = jogos_do_dia[flt]
df = drop_reset_index(df)
df

placares = df[['Odd_0x0','Odd_0x1','Odd_0x2','Odd_0x3',
'Odd_1x0','Odd_1x1','Odd_1x2','Odd_1x3',
'Odd_2x0','Odd_2x1','Odd_2x2','Odd_2x3',
'Odd_3x0','Odd_3x1','Odd_3x2','Odd_3x3',
'Odd_Goleada_H','Odd_CS_Goleada_A','Odd_CS_Goleada_D']]
placares.columns = ['0x0','0x1','0x2','0x3',
'1x0','1x1','1x2','1x3',
'2x0','2x1','2x2','2x3',
'3x0','3x1','3x2','3x3',
'Goleada_H','CS_Goleada_A','CS_Goleada_D']

df1 = placares.T

df2 = df1.sort_values(by=df1.columns[0])

df0 = df2.reset_index()
df0.columns = ['Placar','Odd']
df0 = drop_reset_index(df0)
df0 = df0.head(12)
df0

Probabilidades = [1/Odd for Odd in df0['Odd']]
Soma_Probabilidades = sum(Probabilidades)
Responsabilidade = [(1 / (Odd * Soma_Probabilidades)) * 100 for Odd in df0['Odd']]

df0['Responsabilidade'] = Responsabilidade

df0['Lucro Potencial'] = df0['Odd'] * df0['Responsabilidade'] - 100

print(df0)