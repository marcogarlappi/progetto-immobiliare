def info_dataset(df) -> dict:
    """
    Restituisce un dizionario con le informazioni di base del dataset :
    - numero_righe : int
    - numero_colonne : int
    - colonne : list [ str]
    - tipi_dati : dict ( nome_colonna : tipo )
    - valori_nulli : dict ( nome_colonna : conteggio )
    - percentuale_nulli : dict ( nome_colonna : percentuale )
    """

    info = {
        'numero_righe': df.shape[0],
        'numero_colonne': df.shape[1],
        'colonne': df.columns.tolist(),
        'tipi_dati': df.dtypes.to_dict(),
        'valori_nulli': df.isnull().sum().to_dict(),
        'percentuale_nulli': (df.isnull().mean() * 100).to_dict()
    }
    return info


def gestisci_valori_nulli(df, strategia: str = "media") -> "DataFrame":
    """
    Gestisce i valori nulli nel DataFrame.

    Args :
    df: DataFrame di input
    strategia : " media ", " mediana ", " elimina " o " zero "

    Returns :
    DataFrame pulito

    Raises :
    ValueError : se la strategia non e tra quelle supportate
    """
    if strategia == "media":
        return df.fillna(df.mean())
    elif strategia == "mediana":
        return df.fillna(df.median())
    elif strategia == "elimina":
        return df.dropna()
    elif strategia == "zero":
        return df.fillna(0)
    else:
        raise ValueError(f"Strategia non supportata: {strategia}")


def rileva_outlier(df, colonna: str, metodo: str = "iqr") -> list:
    """
    Rileva gli outlier in una colonna specifica .

    Args :
    df: DataFrame
    colonna : nome della colonna da analizzare
    metodo : "iqr" ( InterQuartile Range ) o " zscore "

    Returns :
    Lista degli indici delle righe con outlier
    """

    if metodo == "iqr":
        Q1 = df[colonna].quantile(0.25)
        Q3 = df[colonna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[colonna] < lower_bound) | (df[colonna] > upper_bound)].index.tolist()
    elif metodo == "zscore":
        mean = df[colonna].mean()
        std = df[colonna].std()
        z_scores = (df[colonna] - mean) / std
        outliers = df[abs(z_scores) > 3].index.tolist()
    else:
        raise ValueError(f"Metodo non supportato: {metodo}")

    return outliers


def normalizza_colonne(df, colonne: list, metodo: str = "minmax"):
    """
    Normalizza le colonne specificate.

    Args :
    df: DataFrame
    colonne : lista di nomi delle colonne
    metodo : "minmax" o "standard" (z- score)

    Returns :
    DataFrame con colonne normalizzate
    """

    df_normalizzato = df.copy()
    for colonna in colonne:
        if metodo == "minmax":
            min_val = df[colonna].min()
            max_val = df[colonna].max()
            df_normalizzato[colonna] = (df[colonna] - min_val) / (max_val - min_val)
        elif metodo == "standard":
            mean = df[colonna].mean()
            std = df[colonna].std()
            df_normalizzato[colonna] = (df[colonna] - mean) / std
        else:
            raise ValueError(f"Metodo non supportato: {metodo}")
    return df_normalizzato
