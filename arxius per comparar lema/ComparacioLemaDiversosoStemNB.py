import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer
import time

# Funció per aplicar lematització i stemming
def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def apply_stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Configurar stemmers i lematitzador
stemmers = {
    "PorterStemmer": PorterStemmer(),
    "SnowballStemmer": SnowballStemmer("english"),
    "RegexpStemmer": RegexpStemmer("ing$|ed$|s$", min=4)
}

# Funció per executar el model amb diferents preprocessaments
def run_naive_bayes_no_clean(data, preprocess_method, method_name):
    # Aplicar el preprocessament especificat
    start = time.time()
    if preprocess_method == "lemmatization":
        data['Tweet'] = data['Tweet'].apply(apply_lemmatization)
    else:
        stemmer = preprocess_method
        data['Tweet'] = data['Tweet'].apply(lambda x: apply_stemming(x, stemmer))
    
    preprocess_time = time.time() - start
    
    # Divisió de dades
    X = data['Tweet']
    y = data['Ranking']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorització amb TF-IDF ajustat
    tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1, 5), min_df=15, max_df=0.9, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    # Entrenar Naive Bayes amb alpha ajustat
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Predicció
    y_pred = model.predict(X_val_tfidf)
    
    # Avaluació
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    return {
        "method": method_name,
        "accuracy": accuracy,
        "preprocess_time": preprocess_time,
        "report": report,
        "conf_matrix": conf_matrix
    }

# Ruta del dataset
dataset_path = 'netejat.csv'  # Si està en un altre lloc, canvia la ruta aquí.

# Intentar carregar el dataset
try:
    dataset_netejat = pd.read_csv(dataset_path)
    print("Dataset carregat correctament!")
except FileNotFoundError:
    print(f"No s'ha trobat el fitxer {dataset_path}. Assegura't que es troba a la ubicació especificada.")
    exit()

# Executar proves amb el dataset netejat i preprocessaments
results_no_clean = []

# Lematització
result = run_naive_bayes_no_clean(dataset_netejat.copy(), "lemmatization", "WordNetLemmatizer")
result["dataset"] = "netejat"
results_no_clean.append(result)

# Stemming
for stemmer_name, stemmer in stemmers.items():
    result = run_naive_bayes_no_clean(dataset_netejat.copy(), stemmer, stemmer_name)
    result["dataset"] = "netejat"
    results_no_clean.append(result)

# Mostrar resultats
for result in results_no_clean:
    print(f"Dataset: {result['dataset']}, Preprocessament: {result['method']}")
    print(f"Exactitud: {result['accuracy'] * 100:.2f}%")
    print(f"Temps de preprocessament: {result['preprocess_time']:.2f} segons")
    print(f"Informe de classificació:\n{result['report']}")
    print(f"Matriu de confusió:\n{result['conf_matrix']}\n")
