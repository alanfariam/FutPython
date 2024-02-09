import pandas as pd

seasons = ["2122", "2223", "2324"]
leagues = ["E0", "E1", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2"]

# Lista para armazenar os Dataframes
dfs = []



# Iterando por cada Temporada e Liga
for season in seasons:
    for league in leagues:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        try:
            df = pd.read_csv(url)
            dfs.append(df)
        except:
            print(f"Erro ao ler a URL: {url}")

# Concatenar todos os Dataframes
df = pd.concat(dfs, ignore_index=True)



df = df[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A", "B365>2.5","B365<2.5"]]
df.columns = ["League", "Date", "Home", "Away", "Goals_H", "Goals_A", "Result", "Odd_H", "Odd_D", "Odd_A", "Odd_Over25", "Odd_Under2.5"]

league_mapping = {
    "E0": "England 1",
    "E1": "England 2",
    "SP1": "Spain 1",
    "SP2": "Spain 2",
    "D1": "Germany 1",
    "D2": "Germany 2",
    "I1": "Italy 1",
    "I2": "Italy 2",
    "F1": "France 1",
    "F2": "France 2"
}
df["League"] = df["League"].replace(league_mapping)


df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
df["Date"] = df["Date"].dt.date
df = df.sort_values(by="Date")

df = df.reset_index(drop=True)
df.index += 1
df

df.to_excel("Base_de_Dados_Football_Data.xlsx",index=False)

print("***FIM***")

