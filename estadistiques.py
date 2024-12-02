import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Llegeix el dataset amb la codificació 'utf-8'
df = pd.read_csv('netejat.csv', encoding='utf-8')

# Estadístiques sobre la llargada dels tweets
tweet_lengths = df['Tweet'].apply(len)
avg_length = tweet_lengths.mean()
min_length = tweet_lengths.min()
max_length = tweet_lengths.max()
std_length = tweet_lengths.std()

# Mostrar els resultats
print(f"Llargada mitjana del tweet: {avg_length} caràcters")
print(f"Llargada mínima del tweet: {min_length} caràcters")
print(f"Llargada màxima del tweet: {max_length} caràcters")
print(f"Desviació típica de la llargada dels tweets: {std_length}")

# Tweet més curt i més llarg
shortest_tweet = df.loc[tweet_lengths.idxmin()]['Tweet']
longest_tweet = df.loc[tweet_lengths.idxmax()]['Tweet']

print(f"Tweet més curt: {shortest_tweet}")
print(f"Tweet més llarg: {longest_tweet}")

# Comptar quants tweets tenen valor 0 i 4 en la columna 'Ranking'
value_0_count = df[df['Ranking'] == 0].shape[0]
value_4_count = df[df['Ranking'] == 4].shape[0]

print(f"Nombre de tweets amb valor 0: {value_0_count}")
print(f"Nombre de tweets amb valor 4: {value_4_count}")

# Crear gràfiques de la llargada dels tweets
plt.figure(figsize=(10,6))
sns.histplot(tweet_lengths, kde=True, color='blue', bins=50)
plt.title('Distribució de la llargada dels Tweets')
plt.xlabel('Llargada del Tweet (caràcters)')
plt.ylabel('Nombre de Tweets')
plt.show()

# Crear gràfica per la distribució de sentiments
plt.figure(figsize=(8,6))
sns.countplot(x='Ranking', data=df, palette='coolwarm')
plt.title('Distribució de Sentiments (0 i 4)')
plt.xlabel('Sentiment')
plt.ylabel('Nombre de Tweets')
plt.show()
