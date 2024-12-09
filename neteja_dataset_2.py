import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Descarregar les stopwords si no estan disponibles
nltk.download('stopwords')

# Llista d'stopwords
stop_words = set(stopwords.words('english'))

# Carregar el dataset netejat
df = pd.read_csv('netejat_1.csv')

# Funció per eliminar stopwords
def remove_stopwords(text):
    # Filtra paraules que no siguin stopwords
    return ' '.join([word for word in text.split() if word not in stop_words])

# Eliminar stopwords de la columna 'Text'
df['Text'] = df['Text'].apply(remove_stopwords)

# Eliminar files on 'Text' està buit després de la neteja
df = df[df['Text'].str.strip() != '']

# Guarda el nou dataset netejat sense stopwords
df.to_csv('netejat_2.csv', index=False)

# Mostra les primeres files per comprovar
print(df.head())
