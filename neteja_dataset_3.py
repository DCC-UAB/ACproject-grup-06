import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk

# Descarregar recursos necessaris de NLTK
nltk.download('wordnet')

# Inicialitzar el WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Carregar el dataset netejat
df = pd.read_csv('netejat_2.csv')

# Funció per aplicar el WordNetLemmatizer
def apply_lemmatizer(text):
    # Aplica lematització a cada paraula del text
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Aplicar la funció de lematització a la columna 'Text'
df['Text'] = df['Text'].apply(apply_lemmatizer)

# Eliminar files on el text estigui buit després de la lematització
df = df[df['Text'].str.strip() != '']

# Guardar el nou dataset netejat
df.to_csv('netejat_3.csv', index=False)

# Mostra les primeres files per comprovar
print(df.head())
