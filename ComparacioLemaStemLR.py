import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Descarregar recursos necessaris
nltk.download('wordnet')

# Configurar stemmers i lematitzador
stemmers = {
    "PorterStemmer": PorterStemmer(),
    "SnowballStemmer": SnowballStemmer("english"),
    "RegexpStemmer": RegexpStemmer("ing$|ed$|s$", min=4),
}

lemmatizer = WordNetLemmatizer()

# Carregar el dataset
df = pd.read_csv('netejat_2.csv')  # Assegura't que el dataset està netejat i sense stopwords
X = df['Text']
y = df['Target']

# Divisió de dades
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Funcions de preprocessament
def apply_stemmer(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def apply_lemmatizer(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Funció per entrenar i avaluar el model
def train_and_evaluate(X_train, X_val, y_train, y_val):
    # Vectorització amb TF-IDF
    tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 4), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # Entrenar i avaluar amb Regressió Logística
    model = LogisticRegression(solver='lbfgs', max_iter=500, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)

    # Mètriques
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    return accuracy, report

# Comparar stemmers i lematitzador
results = {}

for name, stemmer in stemmers.items():
    print(f"Aplicant {name}...")
    X_train_stemmed = X_train.apply(lambda text: apply_stemmer(text, stemmer))
    X_val_stemmed = X_val.apply(lambda text: apply_stemmer(text, stemmer))
    accuracy, report = train_and_evaluate(X_train_stemmed, X_val_stemmed, y_train, y_val)
    results[name] = accuracy
    print(f"Exactitud amb {name}: {accuracy * 100:.2f}%")
    print(report)

# Afegir WordNetLemmatizer
print("Aplicant WordNetLemmatizer...")
X_train_lemmatized = X_train.apply(apply_lemmatizer)
X_val_lemmatized = X_val.apply(apply_lemmatizer)
accuracy, report = train_and_evaluate(X_train_lemmatized, X_val_lemmatized, y_train, y_val)
results["WordNetLemmatizer"] = accuracy
print(f"Exactitud amb WordNetLemmatizer: {accuracy * 100:.2f}%")
print(report)

# Mostrar resultats comparatius
print("\nResultats comparatius:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")
