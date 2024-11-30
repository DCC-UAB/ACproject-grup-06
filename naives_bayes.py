# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar el fitxer
fitxer = 'netejat.csv'
df = pd.read_csv(fitxer, encoding='latin-1')

# Divisió de dades
X = df['Tweet']
y = df['Ranking']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Divisió de les dades fetes")


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
report = classification_report(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("Exactitud del model:")
print(f"{accuracy * 100:.2f}%")
print("Informe de classificació:")
print(report)
print("Matriu de confusió:")
print(conf_matrix)