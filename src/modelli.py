from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.valutazione import cross_validation_modello


def dividi_dataset(df, colonna_target: str,
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Divide il dataset in training set e test set.

    Args:
    df: DataFrame completo
    colonna_target: nome della colonna target
    test_size: proporzione del test set ( default 0.2)
    random_state: seed per la riproducibilita

    Returns:
    tuple: ( X_train, X_test, y_train, y_test )

    Raises:
    ValueError: se test_size non e tra 0 e 1
    KeyError: se colonna_target non esiste nel DataFrame
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
    modello = LinearRegression()
    # Eseguiamo la CV solo per avere una stima della performance sul training set
    risultati_cv = cross_validation_modello(modello, X_train, y_train)

    modello.fit(X_train, y_train)
    return {
        'modello': modello,
        'predizioni': modello.predict(X_test),
        'coefficienti': modello.coef_,
        'intercetta': modello.intercept_,
        'cv_stats': risultati_cv
    }


def addestra_knn(X_train, y_train, X_test, k_list=[3, 5, 7, 9, 11]) -> dict:
    miglior_mse = float('inf')
    miglior_k = None
    miglior_modello = None

    for k in k_list:
        modello = KNeighborsRegressor(n_neighbors=k)
        cv_res = cross_validation_modello(modello, X_train, y_train)

        if cv_res['media'] < miglior_mse:
            miglior_mse = cv_res['media']
            miglior_k = k
            miglior_modello = modello

    # Addestramento finale con il miglior k trovato
    miglior_modello.fit(X_train, y_train)
    return {
        'modello': miglior_modello,
        'predizioni': miglior_modello.predict(X_test),
        'miglior_k': miglior_k,
        'mse_minimo': miglior_mse
    }


def addestra_decision_tree(X_train, y_train, X_test, max_depth_list=[3, 5, 7, 10, None]) -> dict:
    miglior_mse = float('inf')
    miglior_depth = None

    for depth in max_depth_list:
        modello = DecisionTreeRegressor(max_depth=depth, random_state=42)
        cv_res = cross_validation_modello(modello, X_train, y_train)

        if cv_res['media'] < miglior_mse:
            miglior_mse = cv_res['media']
            miglior_depth = depth

    # Addestramento finale
    modello_finale = DecisionTreeRegressor(max_depth=miglior_depth, random_state=42)
    modello_finale.fit(X_train, y_train)

    return {
        'modello': modello_finale,
        'predizioni': modello_finale.predict(X_test),
        'miglior_profondita': miglior_depth,
        'importanza_feature': modello_finale.feature_importances_
    }


def addestra_svr(X_train, y_train, X_test, kernel_list=['linear', 'rbf']) -> dict:
    miglior_mse = float('inf')
    miglior_kernel = None

    for k in kernel_list:
        modello = SVR(kernel=k)
        cv_res = cross_validation_modello(modello, X_train, y_train)

        if cv_res['media'] < miglior_mse:
            miglior_mse = cv_res['media']
            miglior_kernel = k

    modello_finale = SVR(kernel=miglior_kernel)
    modello_finale.fit(X_train, y_train)

    return {
        'modello': modello_finale,
        'predizioni': modello_finale.predict(X_test),
        'miglior_kernel': miglior_kernel
    }


def addestra_tutti_i_modelli(X_train, y_train, X_test) -> dict:
    risultati = {
        "Linear Regression": addestra_regressione_lineare(X_train, y_train, X_test),
        "KNN": addestra_knn(X_train, y_train, X_test),
        "Decision Tree": addestra_decision_tree(X_train, y_train, X_test),
        "SVR": addestra_svr(X_train, y_train, X_test)
    }
    return risultati
