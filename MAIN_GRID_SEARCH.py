import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Funció per carregar i preprocessar les dades
def carregar_dades():
    print("Introdueix el nom del fitxer CSV que vols utilitzar (amb extensió '.csv'):")
    file_path = input("Nom del fitxer: ")
    
    try:
        print("Carregant el dataset...")
        data = pd.read_csv(file_path, encoding='latin-1')
        print("Dataset carregat correctament!")
    except FileNotFoundError:
        print(f"Error: El fitxer '{file_path}' no s'ha trobat. Torna-ho a intentar.")
        return carregar_dades()
    
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
def executar_grid_search(model, param_grid, X_train_tfidf, y_train):
    print(f"Executant Grid Search per {model.__class__.__name__}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Millors hiperparàmetres per {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Millor exactitud: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

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
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
            best_model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
        elif opcio == '2':
            model = MultinomialNB()
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10]
            }
            best_model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
        elif opcio == '3':
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None]
            }
            best_model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
        elif opcio == '4':
            model = SVC(probability=True, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
            best_model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
        elif opcio == '5':
            print("Sortint del programa. Adéu!")
            break
        else:
            print("Opció no vàlida. Torna-ho a intentar.")
            continue

        print("Predint...")
        y_pred = best_model.predict(X_val_tfidf)
        print("Avaluant el model...")
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        print(f"Exactitud del model: {accuracy * 100:.2f}%")
        print("Informe de classificació:")
        print(report)

if __name__ == "__main__":
    main()