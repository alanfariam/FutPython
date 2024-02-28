import pandas as pd
import numpy as np

#importa a base
df = pd.read_csv("https://www.football-data.co.uk/mmz4281/2324/E0.csv")
df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','B365H','B365D','B365A']]
df.columns = ['Date','Home','Away','Goals_H','Goals_A','Odd_H','Odd_D','Odd_A']

print(df)
