import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk

nltk.download('wordnet')


# Carregar el dataset netejat
file_path = 'netejat.csv'
data = pd.read_csv(file_path)

# Divisió de dades (sense filtre, treballant amb totes les dades disponibles)
X = data['Tweet']
y = data['Ranking']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear objectes lemmatizer i stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Funció per aplicar lematització
def apply_lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Funció per aplicar stemming
def apply_stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Vectorització amb TF-IDF
tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 4),
    min_df=8,
    max_df=0.8,
    sublinear_tf=True
)

# Provar amb lematització
start_lemmatizer = time.time()
X_train_lemmatized = X_train.apply(apply_lemmatization)
X_train_tfidf_lemmatized = tfidf.fit_transform(X_train_lemmatized)
X_val_tfidf_lemmatized = tfidf.transform(X_val.apply(apply_lemmatization))
model_lemmatizer = LogisticRegression(
    solver='lbfgs',
    max_iter=500,
    C=0.5,
    class_weight='balanced'
)
model_lemmatizer.fit(X_train_tfidf_lemmatized, y_train)
accuracy_lemmatizer = accuracy_score(y_val, model_lemmatizer.predict(X_val_tfidf_lemmatized))
end_lemmatizer = time.time()

# Provar amb stemming
start_stemmer = time.time()
X_train_stemmed = X_train.apply(apply_stemming)
X_train_tfidf_stemmed = tfidf.fit_transform(X_train_stemmed)
X_val_tfidf_stemmed = tfidf.transform(X_val.apply(apply_stemming))
model_stemmer = LogisticRegression(
    solver='lbfgs',
    max_iter=500,
    C=0.5,
    class_weight='balanced'
)
model_stemmer.fit(X_train_tfidf_stemmed, y_train)
accuracy_stemmer = accuracy_score(y_val, model_stemmer.predict(X_val_tfidf_stemmed))
end_stemmer = time.time()

# Comparació de resultats
times = [end_lemmatizer - start_lemmatizer, end_stemmer - start_stemmer]
accuracies = [accuracy_lemmatizer, accuracy_stemmer]
labels = ['Lemmatization', 'Stemming']

# Gràfiques
plt.figure(figsize=(12, 6))

# Gràfica d'exactitud
plt.subplot(1, 2, 1)
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.title('Comparació d\'exactitud')
plt.ylabel('Exactitud')
plt.ylim(0, 1)

# Gràfica de temps
plt.subplot(1, 2, 2)
plt.bar(labels, times, color=['blue', 'green'])
plt.title('Comparació de temps')
plt.ylabel('Temps (segons)')

plt.tight_layout()
plt.show()

# Resultats finals
print(f"Exactitud amb lematització: {accuracy_lemmatizer * 100:.2f}%")
print(f"Exactitud amb stemming: {accuracy_stemmer * 100:.2f}%")
print(f"Temps amb lematització: {times[0]:.2f} segons")
print(f"Temps amb stemming: {times[1]:.2f} segons")
