import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt

# Carregar el dataset complet netejat
file_path = 'netejat.csv'
data = pd.read_csv(file_path)

# Divisió de dades
X = data['Tweet']
y = data['Ranking']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configurar stemmers i lematitzador
stemmers = {
    "PorterStemmer": PorterStemmer(),
    "SnowballStemmer": SnowballStemmer("english"),
    "RegexpStemmer": RegexpStemmer("ing$|ed$|s$", min=4)
}
lemmatizer = WordNetLemmatizer()

# Funcions per aplicar lematització i stemming
def apply_lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def apply_stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Vectorització amb TF-IDF
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 4), min_df=8, max_df=0.8, sublinear_tf=True)

# Mesures
results = {}

# Lematització
start = time.time()
X_train_lemmatized = X_train.apply(apply_lemmatization)
X_train_tfidf_lemmatized = tfidf.fit_transform(X_train_lemmatized)
X_val_tfidf_lemmatized = tfidf.transform(X_val.apply(apply_lemmatization))
model = LogisticRegression(solver='lbfgs', max_iter=500, C=0.5, class_weight='balanced')
model.fit(X_train_tfidf_lemmatized, y_train)
accuracy = accuracy_score(y_val, model.predict(X_val_tfidf_lemmatized))
results["WordNetLemmatizer"] = {"accuracy": accuracy, "time": time.time() - start}

# Stemmers
for name, stemmer in stemmers.items():
    start = time.time()
    X_train_stemmed = X_train.apply(lambda x: apply_stemming(x, stemmer))
    X_train_tfidf_stemmed = tfidf.fit_transform(X_train_stemmed)
    X_val_tfidf_stemmed = tfidf.transform(X_val.apply(lambda x: apply_stemming(x, stemmer)))
    model = LogisticRegression(solver='lbfgs', max_iter=500, C=0.5, class_weight='balanced')
    model.fit(X_train_tfidf_stemmed, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val_tfidf_stemmed))
    results[name] = {"accuracy": accuracy, "time": time.time() - start}

# Mostrar resultats
print("\nResultats:")
for method, metrics in results.items():
    print(f"{method}: Accuracy = {metrics['accuracy'] * 100:.2f}%, Time = {metrics['time']:.2f} seconds")

# Gràfiques
methods = list(results.keys())
accuracies = [results[method]["accuracy"] for method in methods]
times = [results[method]["time"] for method in methods]

plt.figure(figsize=(12, 6))

# Gràfica d'exactitud
plt.subplot(1, 2, 1)
plt.bar(methods, accuracies, color='blue')
plt.title('Comparació d\'exactitud')
plt.ylabel('Exactitud')
plt.ylim(0, 1)

# Gràfica de temps
plt.subplot(1, 2, 2)
plt.bar(methods, times, color='green')
plt.title('Comparació de temps')
plt.ylabel('Temps (segons)')

plt.tight_layout()
plt.show()
