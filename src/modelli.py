from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def dividi_dataset(df, colonna_target: str,
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Divide il dataset in training set e test set.

    Args :
    df: DataFrame completo
    colonna_target : nome della colonna target
    test_size : proporzione del test set ( default 0.2)
    random_state : seed per la riproducibilita

    Returns :
    tuple : ( X_train , X_test , y_train , y_test )

    Raises :
    ValueError : se test_size non e tra 0 e 1
    KeyError : se colonna_target non esiste nel DataFrame
    """

    if not (0 < test_size < 1):
        raise ValueError("test_size deve essere tra 0 e 1.")
    if colonna_target not in df.columns:
        raise KeyError(f"Colonna target '{colonna_target}' non trovata nel DataFrame.")

    X = df.drop(columns=[colonna_target])
    y = df[colonna_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def addestra_regressione_lineare(X_train, y_train, X_test) -> dict:
    """
    Addestra un modello di Regressione Lineare .

    Returns :
    dict con chiavi : ’modello ’, ’predizioni ’, ’ coefficienti ’, ’intercetta ’
    """

    modello = LinearRegression()
    modello.fit(X_train, y_train)
    predizioni = modello.predict(X_test)
    risultati = {
        'modello': modello,
        'predizioni': predizioni,
        'coefficienti': modello.coef_,
        'intercetta': modello.intercept_
    }
    return risultati


def addestra_knn(X_train, y_train, k_list=[3, 5, 7, 9, 11]) -> dict:
    """
    Trova il miglior valore di k usando la Cross-Validation sul Training Set.

    Returns:
        dict con il miglior modello addestrato e i risultati dettagliati.
    """
    risultati_per_k = {}
    miglior_k = k_list[0]
    miglior_mse_cv = float('inf')

    for k in k_list:
        modello = KNeighborsRegressor(n_neighbors=k)

        # Eseguiamo la Cross-Validation (5-fold)
        # 'neg_mean_squared_error' restituisce valori negativi (più è alto, meglio è)
        scores = cross_val_score(modello, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

        # Trasformiamo in MSE positivo (più è basso, meglio è)
        mse_cv = -scores.mean()

        risultati_per_k[k] = {'mse_cv': mse_cv}

        if mse_cv < miglior_mse_cv:
            miglior_mse_cv = mse_cv
            miglior_k = k

    # Ora addestriamo il modello finale con il MIGLIOR k su TUTTO il training set
    modello_finale = KNeighborsRegressor(n_neighbors=miglior_k)
    modello_finale.fit(X_train, y_train)

    return {
        'modello_ottimizzato': modello_finale,
        'miglior_k': miglior_k,
        'mse_validazione': miglior_mse_cv,
        'tutti_i_risultati': risultati_per_k
    }


def addestra_decision_tree(X_train, y_train, X_test,
                           max_depth_list=[3, 5, 7, 10, None]) -> dict:
    """
    Addestra un modello Decision Tree .
    Sperimenta con diverse profondita (3 , 5 , 7 , 10 , None ).

    Returns :
    dict con chiavi : ’modello ’, ’predizioni ’, ’ miglior_profondita ’,
    ’ importanza_feature ’
    """
    miglior_profondita = None
    miglior_mse = float('inf')
    risultati_cv = {}

    for depth in max_depth_list:
        # 1. Inizializza il modello con la profondità corrente
        albero = DecisionTreeRegressor(max_depth=depth, random_state=42)

        # 2. Cross-Validation per trovare l'errore medio senza usare X_test
        scores = cross_val_score(albero, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_medio = -scores.mean()

        risultati_cv[str(depth)] = mse_medio

        # 3. Aggiorna il miglior parametro
        if mse_medio < miglior_mse:
            miglior_mse = mse_medio
            miglior_profondita = depth

    # 4. Addestramento finale con la migliore profondità su TUTTO il training set
    modello_finale = DecisionTreeRegressor(max_depth=miglior_profondita, random_state=42)
    modello_finale.fit(X_train, y_train)

    # 5. Predizioni sul test set (ora che abbiamo scelto il modello)
    predizioni = modello_finale.predict(X_test)

    # 6. Estrazione importanza delle feature (mappata con i nomi delle colonne)
    importanza = dict(zip(X_train.columns, modello_finale.feature_importances_))

    return {
        'modello': modello_finale,
        'predizioni': predizioni,
        'miglior_profondita': miglior_profondita,
        'importanza_feature': importanza
    }


def addestra_svr(X_train, y_train, X_test,
                 kernel: str = " rbf ") -> dict:
    """
    Addestra un modello Support Vector Regression .
    Sperimenta con diversi kernel ( ’ linear ’, ’rbf ’, ’poly ’).

    Returns :
    dict con chiavi : ’modello ’, ’predizioni ’, ’ miglior_kernel ’
    """

    kernel_list = ['linear', 'rbf', 'poly']
    miglior_kernel = kernel_list[0]
    miglior_mse = float('inf')
    risultati_cv = {}

    for k in kernel_list:
        modello = SVR(kernel=k)
        scores = cross_val_score(modello, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_medio = -scores.mean()
        risultati_cv[k] = mse_medio

        if mse_medio < miglior_mse:
            miglior_mse = mse_medio
            miglior_kernel = k

    modello_finale = SVR(kernel=miglior_kernel)
    modello_finale.fit(X_train, y_train)
    predizioni = modello_finale.predict(X_test)

    return {
        'modello': modello_finale,
        'predizioni': predizioni,
        'miglior_kernel': miglior_kernel,
        'risultati_cv': risultati_cv
    }


def addestra_tutti_i_modelli(X_train, y_train, X_test) -> dict:

    """
    Addestra tutti i modelli disponibili e restituisce i risultati .

    Returns :
    dict : { nome_modello : risultati_dict }
    """

    risultati = {}

    # Regressione Lineare
    risultati['regressione_lineare'] = addestra_regressione_lineare(X_train, y_train, X_test)

    # KNN
    risultati['knn'] = addestra_knn(X_train, y_train)

    # Decision Tree
    risultati['decision_tree'] = addestra_decision_tree(X_train, y_train, X_test)

    # SVR
    risultati['svr'] = addestra_svr(X_train, y_train, X_test)

    return risultati
