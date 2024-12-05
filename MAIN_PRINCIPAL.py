# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:11:54 2024

@author: mcasesf
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Funció per carregar i preprocessar les dades
def carregar_dades():
    print("Carregant el dataset...")
    file_path = r'C:\Users\mcasesf\Documents\ModelML\training.1600000.processed.noemoticon.csv'
    columns = [0, 1, 2, 3, 4, 5]
    data = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)

    print("Seleccionant columnes rellevants...")
    data = data[[0, 5]]
    data.columns = ['target', 'text']

    print("Netejant el text...")
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
    
    print("Dividint les dades...")
    X = data['text']
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Funció per vectoritzar les dades
def vectoritzar_dades(X_train, X_val):
    print("Vectoritzant amb TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 4),
        min_df=8,
        max_df=0.8,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    return X_train_tfidf, X_val_tfidf

# Funció per executar un model

def executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val):
    print(f"Entrenant el model {model.__class__.__name__}...")
    model.fit(X_train_tfidf, y_train)
    print("Predint...")
    y_pred = model.predict(X_val_tfidf)

    print("Avaluant el model...")
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    print(f"Exactitud del model: {accuracy * 100:.2f}%")
    print("Informe de classificació:")
    print(report)

# Funció principal

def main():
    print("Carregant dades i preprocessant...")
    X_train, X_val, y_train, y_val = carregar_dades()
    X_train_tfidf, X_val_tfidf = vectoritzar_dades(X_train, X_val)

    while True:
        print("\nSelecciona un model de Machine Learning:")
        print("1. Regressió Logística")
        print("2. Naive Bayes")
        print("3. Random Forest")
        print("4. SVM")
        print("5. Sortir")

        opcio = input("Introdueix el número de la teva opció: ")

        if opcio == '1':
            model = LogisticRegression(solver='lbfgs', max_iter=500, class_weight='balanced')
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
        elif opcio == '2':
            model = MultinomialNB()
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
        elif opcio == '3':
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
        elif opcio == '4':
            model = SVC(kernel='linear', probability=True, random_state=42)
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
        elif opcio == '5':
            print("Sortint del programa. Adéu!")
            break
        else:
            print("Opció no vàlida. Torna-ho a intentar.")

if __name__ == "__main__":
    main()


#C:\Users\mcasesf\Documents\ModelML\training.1600000.processed.noemoticon.csv