import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
        max_features=100000, #Limitem el nombre de termes a tenir en compte
        ngram_range=(1, 4), #Considerem grups de paraules, desde 1 a 4 paraules
        min_df=8, #Minims documents per tenir en compte una paraula
        max_df=0.8, #Màxim percentatge de documents en el que pot sortir una paraula
        sublinear_tf=True #Aplica un escalat logaritmic, termes molt repetits tenen menys pes
    )
    X_train_tfidf = tfidf.fit_transform(X_train) #transformem conjunt entrenament
    X_val_tfidf = tfidf.transform(X_val)  #transformem conjunt test
    return X_train_tfidf, X_val_tfidf

# Funció per executar Grid Search i retornar tots els resultats
def executar_grid_search_i_graficar(model, param_grid, X_train_tfidf, y_train, model_name):
    print(f"Executant Grid Search per {model_name}...")
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=3,
                               scoring='accuracy',
                               verbose=1,
                               return_train_score=True)
    grid_search.fit(X_train_tfidf, y_train)

    print(f"Millors hiperparàmetres per {model_name}: {grid_search.best_params_}")
    print(f"Millor accuracy: {grid_search.best_score_ * 100:.2f}%")

    # Convertir els resultats del GridSearch a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Crear una gràfica per representar les combinacions d'hiperparàmetres
    param_comb_names = [
        ', '.join([f"{key}={value}" for key, value in comb.items()])
        for comb in results_df['params']
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(param_comb_names, results_df['mean_test_score'], color='skyblue')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Accuracy per combinacions d\'hiperparàmetres ({model_name})', fontsize=16)
    plt.xlabel("Combinacions d'hiperparàmetres", fontsize=14)
    plt.ylabel('Accuracy (mitjana de validació)', fontsize=14)
    plt.ylim(0.6, 1.0)  # Escala ajustada per començar a 0.6
    for i, acc in enumerate(results_df['mean_test_score']):
        plt.text(i, acc + 0.005, f'{acc:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'gridsearch_{model_name}_accuracy.png')
    plt.show()

    return grid_search.best_estimator_

# Funció per entrenar i avaluar un model directament
def executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val):
    print(f"Entrenant el model {model.__class__.__name__}...")
    model.fit(X_train_tfidf, y_train) #Utilitzem les dades vectoritzades
    print("Predint...")
    y_pred = model.predict(X_val_tfidf) #Fa la prediccio

    print("Avaluant el model...")
    accuracy = accuracy_score(y_val, y_pred) #obteim mètriques
    print(f"Exactitud del model: {accuracy * 100:.2f}%")
    return accuracy

# Funció principal
def main():
    print("Carregant dades i preprocessant...")
    X_train, X_val, y_train, y_val = carregar_dades()
    X_train_tfidf, X_val_tfidf = vectoritzar_dades(X_train, X_val)

    models = {
        'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=500, class_weight='balanced'),
        'NaiveBayes': MultinomialNB(),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced')
    }

    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'NaiveBayes': {
            'alpha': [0.01, 0.1, 1, 10]
        },
        'RandomForest': {
            'n_estimators': [20, 40],
            'max_depth': [10, 20]
        }
    }

    results = []

    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        print("Optimitzant hiperparàmetres amb Grid Search...")
        model = executar_grid_search_i_graficar(model, param_grids[model_name], X_train_tfidf, y_train, model_name)
        accuracy = executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)
        results.append({'Model': model_name, 'Accuracy': accuracy})

    # Crear una taula amb els resultats finals
    results_df = pd.DataFrame(results)
    print("\nResultats finals:")
    print(results_df)

    # Generar gràfica de comparació d'accuracy final
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df['Accuracy'] * 100, color='skyblue')
    plt.title('Comparació d\'accuracy entre models', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    for i, acc in enumerate(results_df['Accuracy'] * 100):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('accuracy_comparison_final.png')
    plt.show()

if __name__ == "__main__":
    main()
