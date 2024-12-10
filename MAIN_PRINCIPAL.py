##CODI PRINCIPAL ON S'EXECUTEN TOTS ELS MODELS DE ML EN UN SOL MAIN, CAL MODIFICAR PER IMPORTAR UN DATASET JA NETEJAT.
##Implementacions futures --> Posar un input per demanar quin CSV importar.

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
        max_features=100000, #Limitem el nombre de termes a tenir en compte
        ngram_range=(1, 4), #Considerem grups de paraules, desde 1 a 4 paraules
        min_df=8, #Minims documents per tenir en compte una paraula
        max_df=0.8, #Màxim percentatge de documents en el que pot sortir una paraula
        sublinear_tf=True #Aplica un escalat logaritmic, termes molt repetits tenen menys pes
    )
    X_train_tfidf = tfidf.fit_transform(X_train) #transformem conjunt entrenament
    X_val_tfidf = tfidf.transform(X_val)  #transformem conjunt test
    return X_train_tfidf, X_val_tfidf

# Funció per executar Grid Search
def executar_grid_search(model, param_grid, X_train_tfidf, y_train):
    print(f"Executant Grid Search per {model._class.name_}...")
    print("Hiperparàmetres que s'optimitzaran:")
    for param, values in param_grid.items():
        print(f" - {param}: {values}")
    

    grid_search = GridSearchCV(estimator=model, #Indiquem el model
                               param_grid=param_grid, #Indiquem el diccionari de hiperparàmetres
                               cv=3, #Cross-validation amb 3 particions
                               scoring='accuracy', #optimitzem per a accuracy
                               verbose=1) #Ens mostra info detallada
    grid_search.fit(X_train_tfidf, y_train)
    print(f"Millors hiperparàmetres per {model._class.name}: {grid_search.best_params}")
    print(f"Millor exactitud: {grid_search.best_score_ * 100:.2f}%")
    return grid_search.best_estimator_

# Funció per entrenar i avaluar un model directament
def executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val):
    print(f"Entrenant el model {model._class.name_}...")
    model.fit(X_train_tfidf, y_train) #Utilitzem les dades vectoritzades
    print("Predint...")
    y_pred = model.predict(X_val_tfidf) #Fa la prediccio

    print("Avaluant el model...")
    accuracy = accuracy_score(y_val, y_pred) #obteim mètriques
    report = classification_report(y_val, y_pred) #obtenim mètriques
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

        if opcio == '5':
            print("Sortint del programa. Adéu!")
            break

        print("Vols fer servir Grid Search per optimitzar els hiperparàmetres? (s/n)")
        fer_grid_search = input().lower()

        if opcio == '1':
            model = LogisticRegression(solver='lbfgs', #Alorisme d'optimitzacio per trobar els pesos del model(adecuat per datasets grans)
                                       max_iter=500, #Max d'itercions per assegurar convergencia
                                       class_weight='balanced') #Ajusta pes de les classes segons distribucio
            param_grid = {
                'C': [0.01, 0.1, 1, 10],#per evitar sobreajustamet. aplica una pealitzacio a caracteristiques massa específiques. Quant més gran, menys força a la regularització
                'solver': ['lbfgs', 'liblinear'] #Libliear seria adecuat per a datasets petits
            }
            if fer_grid_search == 's':
                model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)

        elif opcio == '2':
            model = MultinomialNB()
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10]
            }
            if fer_grid_search == 's':
                model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)

        elif opcio == '3':
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None]
            }
            if fer_grid_search == 's':
                model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)

        elif opcio == '4':
            model = SVC(probability=True, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
            if fer_grid_search == 's':
                model = executar_grid_search(model, param_grid, X_train_tfidf, y_train)
            executar_model(model, X_train_tfidf, X_val_tfidf, y_train, y_val)

        else:
            print("Opció no vàlida. Torna-ho a intentar.")

if __name__ == "__main__":
    main()
