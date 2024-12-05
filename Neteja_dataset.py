import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Carrega el dataset amb un encoding diferent (latin1 o ISO-8859-1)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')

# Assigna noms a les columnes
df.columns = ['Target', 'Timestamp', 'Date', 'Query_Type', 'Username', 'Text']

# Funció per netejar el tweet (eliminar caràcters no desitjats com emojis o símbols)
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Eliminar enllaços
    text = re.sub(r'@\w+', '', text)  # Eliminar mencions
    text = re.sub(r'#', '', text)  # Eliminar #
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuació
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = text.lower()  # Minúscules
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Neteja els tweets i crea una nova columna 'Cleaned_Tweet'
df['Text'] = df['Text'].apply(clean_text)

# Queda't només amb les columnes 'Ranking' i 'Tweet'
df_cleaned = df[['Target', 'Text']]

# Guarda el nou dataset netejat
df_cleaned.to_csv('netejat.csv', index=False)

# Mostra les primeres files per comprovar
print(df_cleaned.head())


