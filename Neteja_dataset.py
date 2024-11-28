import pandas as pd

#fitxer que deixa només les columnes de ranking i de tweet

# Carrega el dataset amb un encoding diferent (latin1 o ISO-8859-1)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')

# Assigna noms a les columnes
df.columns = ['Ranking', 'Timestamp', 'Date', 'Query_Type', 'Username', 'Tweet']

# Filtra les files amb 'Ranking' 0 o 4
df_filtered = df[df['Ranking'].isin([0, 4])]

# Queda't només amb les columnes 'Ranking' i 'Tweet'
df_cleaned = df_filtered[['Ranking', 'Tweet']]

# Guarda el nou dataset netejat
df_cleaned.to_csv('netejat.csv', index=False)

# Mostra les primeres files per comprovar
print(df_cleaned.head())
