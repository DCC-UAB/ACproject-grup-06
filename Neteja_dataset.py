import pandas as pd
import re
#fitxer que deixa només les columnes de ranking i de tweet

# Carrega el dataset amb un encoding diferent (latin1 o ISO-8859-1)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')

# Assigna noms a les columnes
df.columns = ['Ranking', 'Timestamp', 'Date', 'Query_Type', 'Username', 'Tweet']

# Funció per netejar el tweet (eliminar caràcters no desitjats com emojis o símbols)
def clean_tweet(text):
    # Eliminem tot el que no sigui lletres, números o espais
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Eliminem espais al principi i al final
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Neteja els tweets i crea una nova columna 'Cleaned_Tweet'
df['Tweet'] = df['Tweet'].apply(clean_tweet)

# Queda't només amb les columnes 'Ranking' i 'Tweet'
df_cleaned = df[['Ranking', 'Tweet']]

# Guarda el nou dataset netejat
df_cleaned.to_csv('netejat.csv', index=False)

# Mostra les primeres files per comprovar
print(df_cleaned.head())
