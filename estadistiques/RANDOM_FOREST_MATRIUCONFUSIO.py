# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:33:40 2024

@author: mcasesf
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Funció per carregar i preprocessar les dades
def carregar_dades(file_path):
    try:
        print(f"Carregant el dataset: {file_path}...")
        data = pd.read_csv(file_path, encoding='latin-1')
        print("Dataset carregat correctament!")
    except FileNotFoundError:
        print(f"Error: El fitxer '{file_path}' no s'ha trobat. Torna-ho a intentar.")
        return None, None, None, None
    
    print("Dividint les dades...")
    X = data['Text']
    y = data['Target']
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

# Funció per executar Grid Search
def executar_grid_search(X_train_tfidf, y_train):
    print("Executant Grid Search per Random Forest...")
    param_grid = {
        'n_estimators': [20, 40],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Millors hiperparàmetres: {grid_search.best_params_}")
    print(f"Millor exactitud: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

# Funció principal
def main():
    dataset = r'C:\Users\mcasesf\Documents\ModelML\netejat_1.csv'
    output_dir = r'C:\Users\mcasesf\Documents\ModelML'
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_val, y_train, y_val = carregar_dades(dataset)

    if X_train is None:
        return

    X_train_tfidf, X_val_tfidf = vectoritzar_dades(X_train, X_val)

    print("Optimitzant hiperparàmetres amb Grid Search...")
    model = executar_grid_search(X_train_tfidf, y_train)
    
    print("Predint...")
    y_pred = model.predict(X_val_tfidf)

    print("Generant la matriu de confusió...")
    cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriu de Confusió - Random Forest (Grid Search)')
    plt.xlabel('Prediccions')
    plt.ylabel('Valors Reals')
    plt.savefig(os.path.join(output_dir, 'matriu_confusio_random_forest_grid_search.png'))
    plt.show()

if __name__ == "__main__":
    main()
