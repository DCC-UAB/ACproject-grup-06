import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Funció per carregar i preprocessar les dades
def carregar_dades(file_path):
    try:
        print(f"Carregant el dataset: {file_path}...")
        data = pd.read_csv(file_path, encoding='latin-1')
        print("Dataset carregat correctament!")
    except FileNotFoundError:
        print(f"Error: El fitxer '{file_path}' no s'ha trobat. Torna-ho a intentar.")
        return None
    
    print("Dividint les dades...")
    X = data['Text']
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Funció per vectoritzar les dades
def vectoritzar_dades(X_train, X_val):
    print("Vectoritzant amb TF-IDF...")
    tfidf = TfidfVectorizer(  # Configura el vectoritzador.
        max_features=100000,  # Limita el nombre màxim de característiques.
        ngram_range=(1, 4),  # Utilitza un rang d'un a quatre grams. Grups de paraules
        min_df=8,  # Exclou termes que apareixen en menys de 8 documents.
        max_df=0.8,  # Exclou termes que apareixen en més del 80% dels documents.
        sublinear_tf=True  # Aplica una transformació sublineal al TF.
    )
    X_train_tfidf = tfidf.fit_transform(X_train) # Ajusta i transforma les dades d'entrenament.
    X_val_tfidf = tfidf.transform(X_val) # Transforma les dades de validació
    return X_train_tfidf, X_val_tfidf

# Funció per entrenar i avaluar un model directament
def executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val):
    print(f"Entrenant el model {model.__class__.__name__}...")
    model.fit(X_train_tfidf, y_train)
    print("Predint...")
    y_pred = model.predict(X_val_tfidf)

    print("Avaluant el model...")
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Exactitud del model: {accuracy * 100:.2f}%")
    return accuracy

# Funció per executar Grid Search
def executar_grid_search(model, param_grid, X_train_tfidf, y_train):
    print(f"Executant Grid Search per {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Millors hiperparàmetres per {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Millor exactitud: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

# Funció principal
def main():
    datasets = ['Datasets/netejat_1.csv', 'Datasets/netejat_2.csv', 'Datasets/netejat_3.csv']  # Afegeix els noms dels teus datasets
    models = {
        'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=500, class_weight='balanced'),
        'NaiveBayes': MultinomialNB(),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42)
    }

    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'NaiveBayes': {
            'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5],
            'kernel': ['linear', 'rbf']
        }
    }

    results = {}

    for dataset in datasets:
        X_train, X_val, y_train, y_val = carregar_dades(dataset)
        X_train_tfidf, X_val_tfidf = vectoritzar_dades(X_train, X_val)

        for model_name, model in models.items():
            print(f"\nModel: {model_name} | Dataset: {dataset}")
            print(f"Optimitzant hiperparàmetres per {model_name} amb Grid Search...")
            model = executar_grid_search(model, param_grids[model_name], X_train_tfidf, y_train)

            accuracy = executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
            
            if model_name not in results:
                results[model_name] = {}
            results[model_name][dataset] = accuracy

    # Crear un DataFrame amb els resultats
    results_df = pd.DataFrame(results).T  # Transposar perquè models siguin files i datasets columnes
    print("\nResultats finals:")
    print(results_df)

    # Guardar els resultats en un fitxer CSV
    results_df.to_csv('resultats_models.csv', index=True)

if __name__ == "__main__":
    main()
