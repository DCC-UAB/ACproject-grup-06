import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk

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
    return text

# Funció per eliminar stopwords
def remove_stopwords(text):
    # Filtra paraules que no siguin stopwords
    return ' '.join([word for word in text.split() if word not in stop_words])

# Funció per aplicar el WordNetLemmatizer
def apply_lemmatizer(text):
    # Aplica lematització a cada paraula del text
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Funció per aplicar SnowballStemmer
def apply_stemming(text):
    # Aplica stemming a cada paraula del text
    return ' '.join([stemmer.stem(word) for word in text.split()])

opcio = int(input("\nQuin CSV vols crear? 1, 2, 3, 4: "))

if opcio == 1:
    #Carregar el dataset amb un encoding diferent (latin1)
    df = pd.read_csv('Datasets/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
    #Assigna noms a les columnes
    df.columns = ['Target', 'Timestamp', 'Date', 'Query_Type', 'Username', 'Text']
    # Queda't només amb les columnes 'Target' i 'Text'
    df = df[['Target', 'Text']]
    # Neteja els tweets sobre la columna 'Text'
    df['Text'] = df['Text'].apply(clean_text)
    # Eliminar files on 'Text' està buit després de la neteja
    df = df[df['Text'].str.strip() != '']
    # Guarda el nou dataset netejat
    df.to_csv('Datasets/netejat_1.csv', index=False)

if opcio == 2:
    # Descarregar les stopwords si no estan disponibles
    nltk.download('stopwords')
    # Llista d'stopwords
    stop_words = set(stopwords.words('english'))
    # Carregar el dataset netejat
    df = pd.read_csv('Datasets/netejat_1.csv')
    # Eliminar stopwords de la columna 'Text'
    df['Text'] = df['Text'].apply(remove_stopwords)
    # Eliminar files on 'Text' està buit després de la neteja
    df = df[df['Text'].str.strip() != '']
    # Guarda el nou dataset netejat sense stopwords
    df.to_csv('Datasets/netejat_2.csv', index=False)

if opcio == 3:
    # Descarregar recursos necessaris de NLTK
    nltk.download('wordnet')
    # Inicialitzar el WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Carregar el dataset netejat
    df = pd.read_csv('Datasets/netejat_2.csv')
    # Aplicar la funció de lematització a la columna 'Text'
    df['Text'] = df['Text'].apply(apply_lemmatizer)
    # Eliminar files on el text estigui buit després de la lematització
    df = df[df['Text'].str.strip() != '']
    # Guardar el nou dataset netejat
    df.to_csv('Datasets/netejat_3.csv', index=False)

if opcio == 4:
    # Descarregar recursos necessaris de NLTK
    nltk.download('punkt')
    # Inicialitzar el SnowballStemmer
    stemmer = SnowballStemmer("english")
    # Carregar el dataset netejat
    df = pd.read_csv('Datasets/netejat_2.csv')
    # Aplicar la funció de stemming a la columna 'Text'
    df['Text'] = df['Text'].apply(apply_stemming)
    # Eliminar files on el text estigui buit després de l'stemming
    df = df[df['Text'].str.strip() != '']
    # Guardar el nou dataset netejat
    df.to_csv('Datasets/netejat_4.csv', index=False)

# Mostra les primeres files per comprovar
print(df.head())
