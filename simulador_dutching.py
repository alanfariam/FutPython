import pandas as pd
import numpy as np
from scipy.stats import poisson

import warnings
warnings.filterwarnings("ignore")

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def simulate_match(home_goals_for, home_goals_against, away_goals_for, away_goals_against, num_simulations=10000):
    estimated_home_goals = (home_goals_for + away_goals_against) / 2
    estimated_away_goals = (away_goals_for + home_goals_against) / 2

    home_goals = poisson(estimated_home_goals).rvs(num_simulations)
    away_goals = poisson(estimated_away_goals).rvs(num_simulations)

    results = pd.DataFrame({
        'Home_Goals': home_goals,
        'Away_Goals': away_goals
    })

    return results

def top_results_df(simulated_results, top_n=10):

    result_counts = simulated_results.value_counts().head(top_n).reset_index()
    result_counts.columns = ['Home_Goals', 'Away_Goals', 'Count']

    sum_top_counts = result_counts['Count'].sum()
    result_counts['Probability'] = result_counts['Count'] / sum_top_counts

    return result_counts

df = pd.read_csv("https://www.football-data.co.uk/mmz4281/2223/E0.csv")
df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','B365H','B365D','B365A']]
df.columns = ['Date','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A']
df = drop_reset_index(df)

df_train = df[:-100]
df_test = df[-100:-90]

Team_01 = df_test['Home'][285]
Team_02 = df_test['Away'][285]

Home = df_train[df_train['Home'] == Team_01].tail(5)
Away = df_train[df_train['Away'] == Team_02].tail(5)

# Média de Gols Marcados
Media_GM_H = Home['Goals_H'].mean()
Media_GM_A = Away['Goals_A'].mean()

# Média de Gols Sofridos
Media_GS_H = Home['Goals_A'].mean()
Media_GS_A = Away['Goals_H'].mean()

# Simular Partidas
simulated_results = simulate_match(Media_GM_H, Media_GS_H, Media_GM_A, Media_GS_H)
df = top_results_df(simulated_results)
df = drop_reset_index(df)
#display(df)

# Análise dos resultados
Home = sum(df['Home_Goals'] > df['Away_Goals'])
Draw = sum(df['Home_Goals'] == df['Away_Goals'])
Away = sum(df['Home_Goals'] < df['Away_Goals'])

print(f"Vitórias {Team_01}: {Home/len(df):.2f}")
print(f"Empates: {Draw/len(df):.2f}")
print(f"Vitórias {Team_02}: {Away/len(df):.2f}")