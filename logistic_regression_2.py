# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Carregar el fitxer
file_path = r'C:\Users\mcasesf\Documents\ModelML\training.1600000.processed.noemoticon.csv'
columns = [0, 1, 2, 3, 4, 5]
data = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)

# Seleccionar columnes rellevants
data = data[[0, 5]]
data.columns = ['target', 'text']

# Preprocessament amb lematització
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Eliminar enllaços
    text = re.sub(r'@\w+', '', text)  # Eliminar mencions
    text = re.sub(r'#', '', text)  # Eliminar #
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuació
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = text.lower()  # Minúscules
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

data['text'] = data['text'].apply(clean_text)

# Divisió de dades
X = data['text']
y = data['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorització amb TF-IDF millorada
tfidf = TfidfVectorizer(
    max_features=100000, 
    ngram_range=(1, 4), 
    min_df=8, 
    max_df=0.8, 
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Entrenar regressió logística ajustada
model = LogisticRegression(
    solver='lbfgs', 
    max_iter=500, 
    C=0.5, #Ajuda a reduïr l'overfitting
    class_weight='balanced' #Similar a estandarització  --> Influència classe majoritària
)

model.fit(X_train_tfidf, y_train)

# Predicció
y_pred = model.predict(X_val_tfidf)

# Avaluació
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("Exactitud del model:")
print(f"{accuracy * 100:.2f}%")
print("Informe de classificació:")
print(report)
print("Matriu de confusió:")
print(conf_matrix)

# Cross-validation --> Per a realitzar les diferents combinacions de dades train i test amb k-folds 
scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Mitjana d'exactitud amb cross-validation: {scores.mean() * 100:.2f}%")
