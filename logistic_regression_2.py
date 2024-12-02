# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
"""
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
"""
# Carregar el fitxer
fitxer = 'netejat.csv'
df = pd.read_csv(fitxer, encoding='latin-1')

# Divisió de dades
X = df['Tweet']
y = df['Ranking']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Divisió de les dades fetes")

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
