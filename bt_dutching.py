import pandas as pd
import numpy as np
from scipy.stats import poisson

ligas = [
'ENGLAND - CHAMPIONSHIP',
'ENGLAND - PREMIER LEAGUE',
'FRANCE - LIGUE 1',
'FRANCE - LIGUE 2',
'GERMANY - 2. BUNDESLIGA',
'GERMANY - BUNDESLIGA',
'ITALY - SERIE A',
'ITALY - SERIE B',
'SPAIN - LALIGA',
'SPAIN - LALIGA2'
]

import warnings
warnings.filterwarnings("ignore")

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)
    estimated_home_goals = (home_goals_for + away_goals_against) / 2
    estimated_away_goals = (away_goals_for + home_goals_against) / 2

    home_goals = poisson(estimated_home_goals).rvs(num_simulations)
    away_goals = poisson(estimated_away_goals).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results

def top_results_df(simulated_results, top_n):

    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

def plot_profit_acu(dataframe, title_text):
    dataframe['Profit_acu'] = dataframe.Profit.cumsum()
    n_apostas = dataframe.shape[0]
    profit = round(dataframe.Profit_acu.tail(1).item(), 2)
    ROI = round((dataframe.Profit_acu.tail(1) / n_apostas * 100).item(), 2)
    drawdown = dataframe['Profit_acu'] - dataframe['Profit_acu'].cummax()
    drawdown_maximo = round(drawdown.min(), 2)
    winrate_medio = round((dataframe['Profit'] > 0).mean() * 100, 2)
    desvio_padrao = round(dataframe['Profit'].std(), 2)
    dataframe.Profit_acu.plot(title=title_text, xlabel='Entradas', ylabel='Stakes')
    print("Método:",title_text)
    print("Profit:", profit, "stakes em", n_apostas, "jogos")
    print("ROI:", ROI, "%")
    print("Drawdown Maximo Acumulado:", drawdown_maximo)
    print("Winrate Medio:", winrate_medio, "%")
    print("Desvio Padrao:", desvio_padrao)
    print("")

base = pd.read_csv("https://github.com/futpythontrader/YouTube/raw/main/Base_de_Dados/Base%20de%20Dados_Betfair_Exchange.csv")
base = base[['Date','League','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A',
'CS_0x0','CS_0x1','CS_0x2','CS_0x3',
'CS_1x0','CS_1x1','CS_1x2','CS_1x3',
'CS_2x0','CS_2x1','CS_2x2','CS_2x3',
'CS_3x0','CS_3x1','CS_3x2','CS_3x3',
'CS_Goleada_H','CS_Goleada_A']]
base.columns = ['Date','League','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A',
              'Odd_0x0','Odd_0x1','Odd_0x2','Odd_0x3',
              'Odd_1x0','Odd_1x1','Odd_1x2','Odd_1x3',
              'Odd_2x0','Odd_2x1','Odd_2x2','Odd_2x3',
              'Odd_3x0','Odd_3x1','Odd_3x2','Odd_3x3',
              'Odd_Goleada_H','Odd_Goleada_A']
base = base[base['League'].isin(ligas) == True]
base = drop_reset_index(base)

# Períodos
n = 5

# Média de Gols Marcados
base['Media_GM_H'] = base.groupby('Home')['Goals_H'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GM_A'] = base.groupby('Away')['Goals_A'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)

base['Media_GM_H'] = base.groupby('Home')['Media_GM_H'].shift(1)
base['Media_GM_A'] = base.groupby('Away')['Media_GM_A'].shift(1)

# Média de Gols Sofridos
base['Media_GS_H'] = base.groupby('Home')['Goals_A'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)
base['Media_GS_A'] = base.groupby('Away')['Goals_H'].rolling(window=n, min_periods=n).mean().reset_index(0,drop=True)

base['Media_GS_H'] = base.groupby('Home')['Media_GS_H'].shift(1)
base['Media_GS_A'] = base.groupby('Away')['Media_GS_A'].shift(1)

base = drop_reset_index(base)

base.replace([1.01, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000], np.nan, inplace=True)
base = drop_reset_index(base)
base

'''
#1 jogo

i = 1

df = base

df = df[df.index == i]
pd.set_option('display.max_columns', None)
df

Team_01 = df['Home'][i]
Team_02 = df['Away'][i]

# Média de Gols Marcados
Media_GM_H = df['Media_GM_H'][i]
Media_GM_A = df['Media_GM_A'][i]

# Média de Gols Sofridos
Media_GS_H = df['Media_GS_H'][i]
Media_GS_A = df['Media_GS_A'][i]

# Simular Partidas
simulated_results = simulate_match(Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
simulated_results = drop_reset_index(simulated_results)

print(f'Jogo: {Team_01} x {Team_02}')
simulated_results

# Somando a quantidade de todos os resultados diferentes simulados
results = top_results_df(simulated_results,100)
results = drop_reset_index(results)
results

# Criando as Colunas de Goleadas
results['Placar'] = results.apply(
    lambda row: 'Goleada_H' if (row['Home_Goals'] >= 4 and row['Home_Goals'] > row['Away_Goals'])
    else 'Goleada_A' if (row['Away_Goals'] >= 4 and row['Away_Goals'] > row['Home_Goals'])
    else 'Goleada_D' if (row['Home_Goals'] >= 4 and row['Away_Goals'] >= 4 and row['Home_Goals'] == row['Away_Goals'])
    else f"{int(row['Home_Goals'])}x{int(row['Away_Goals'])}", axis=1
)
results

# Selecionando os 10 resultados mais prováveis
results = results.head(10)
results

# Buscar a odd correspondente no DataFrame original e adicionar essa odd ao DataFrame de resultados
for index, row in results.iterrows():

    odd_column_name = f'Odd_{row["Placar"]}'
    try:
        odd_value = df[odd_column_name][i]
    except:
        pass
    results.at[index, 'Odd'] = odd_value

results

# Calculando a Stake para o Dutching
stake_total = 100

inverso_das_odds = [1/odd for odd in results['Odd']]
soma_inverso_das_odds = sum(inverso_das_odds)
stakes = [(1 / (odd * soma_inverso_das_odds)) * stake_total for odd in results['Odd']]

results['Stake'] = stakes

results['Lucro Potencial'] = results['Odd'] * results['Stake'] - stake_total
results

# Calculando o Valor Esperado
results['EV'] = results['Probability'] * results['Odd'] - 1
results

EV = round((results['EV'].sum()),2)
df.loc[i, 'EV'] = EV
print(f'{EV} %')

# Obtendo o resultado correto e criando a coluna de Green e Red
real_home_goals = df['Goals_H'][i]
real_away_goals = df['Goals_A'][i]

results['Green_Red'] = results.apply(lambda row: 'Green' if (row['Home_Goals'] == real_home_goals and row['Away_Goals'] == real_away_goals) else 'Red', axis=1)
results

# Criando o Profit em cima dos Greens e Reds
if 'Green' in results['Green_Red'].values:
    profit = results.loc[results['Green_Red'] == 'Green', 'Lucro Potencial'].iloc[0]
else:
    profit = -100

df.loc[i, 'Profit'] = profit
df
'''

df = base

for k in range(len(df)):

    i = k + 1

    Team_01 = df['Home'][i]
    Team_02 = df['Away'][i]

    Media_GM_H = df['Media_GM_H'][i]
    Media_GM_A = df['Media_GM_A'][i]

    Media_GS_H = df['Media_GS_H'][i]
    Media_GS_A = df['Media_GS_A'][i]

    simulated_results = simulate_match(Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
    simulated_results = drop_reset_index(simulated_results)

    results = top_results_df(simulated_results,100)
    results = drop_reset_index(results)

    results['Placar'] = results.apply(
        lambda row: 'Goleada_H' if (row['Home_Goals'] >= 4 and row['Home_Goals'] > row['Away_Goals'])
        else 'Goleada_A' if (row['Away_Goals'] >= 4 and row['Away_Goals'] > row['Home_Goals'])
        else 'Goleada_D' if (row['Home_Goals'] >= 4 and row['Away_Goals'] >= 4 and row['Home_Goals'] == row['Away_Goals'])
        else f"{int(row['Home_Goals'])}x{int(row['Away_Goals'])}", axis=1
    )

    results = results.head(10)

    for index, row in results.iterrows():

        odd_column_name = f'Odd_{row["Placar"]}'
        try:
            odd_value = df[odd_column_name][i]
        except:
            pass
        results.at[index, 'Odd'] = odd_value

    stake_total = 1

    inverso_das_odds = [1/odd for odd in results['Odd']]
    soma_inverso_das_odds = sum(inverso_das_odds)
    stakes = [(1 / (odd * soma_inverso_das_odds)) * stake_total for odd in results['Odd']]

    results['Stake'] = stakes

    results['Lucro Potencial'] = results['Odd'] * results['Stake'] - stake_total

    results['EV'] = results['Probability'] * results['Odd'] - 1

    EV = round((results['EV'].sum()),2)
    df.loc[i, 'EV'] = EV

    real_home_goals = df['Goals_H'][i]
    real_away_goals = df['Goals_A'][i]

    results['Green_Red'] = results.apply(lambda row: 'Green' if (row['Home_Goals'] == real_home_goals and row['Away_Goals'] == real_away_goals) else 'Red', axis=1)

    if 'Green' in results['Green_Red'].values:
        profit = results.loc[results['Green_Red'] == 'Green', 'Lucro Potencial'].iloc[0]
    else:
        profit = -1

    df.loc[i, 'Profit'] = profit
    # df.loc[i, 'Profit'] = profit*0.935 # Betfair
    # df.loc[i, 'Profit'] = profit*0.9675 # Bolsa de Aposta

plot_profit_acu(df, 'Dutching CS')

flt = ((df.EV > 1))
df0 = df[flt]
df0 = drop_reset_index(df0)
plot_profit_acu(df0, 'Dutching CS')
