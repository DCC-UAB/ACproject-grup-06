import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import nltk

# Descarregar recursos necessaris de nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Carregar el dataset amb un encoding diferent (latin1)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')

# Assigna noms a les columnes
df.columns = ['Target', 'Timestamp', 'Date', 'Query_Type', 'Username', 'Text']

# Queda't només amb les columnes 'Target' i 'Text'
df = df[['Target', 'Text']]

# Funció per netejar el tweet (eliminar caràcters no desitjats com símbols)
def clean_text(text):
    # Eliminar enllaços
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Eliminar mencions
    text = re.sub(r'@\w+', '', text)
    # Eliminar '#'
    text = re.sub(r'#', '', text)
    # Eliminar puntuació i símbols no alfanumèrics
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    # Convertir a minúscules
    text = text.lower()
    # Lemmatització de cada paraula
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Neteja els tweets sobre la columna 'Text'
df['Text'] = df['Text'].apply(clean_text)

# Eliminar files on 'Text' està buit després de la neteja
df = df[df['Text'].str.strip() != '']

# Guarda el nou dataset netejat
df.to_csv('netejat.csv', index=False)

# Mostra les primeres files per comprovar
print(df.head())


