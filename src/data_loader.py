from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def carica_dataset() -> tuple:
    """
    Carica il California Housing Dataset da scikit - learn .

    Returns :
    tuple : ( DataFrame con le feature , Series con il target )
    """
    data = fetch_california_housing(as_frame=True)
    tuple_data = (data.data, data.target)
    return tuple_data

def salva_csv(df, percorso: str) -> None:
    """
    Salva un DataFrame in formato CSV nella cartella data /.
    Verifica che la cartella esista , altrimenti la crea .

    Args :
    df: DataFrame da salvare


    percorso : percorso del file di output
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('output'):
        os.makedirs('output')
    df.to_csv(percorso, index=False)


def carica_csv(percorso: str):
    """
    Carica un file CSV e lo restituisce come DataFrame .
    Gestisce il caso in cui il file non esista .

    Args :
    percorso : percorso del file da caricare

    Returns :
    DataFrame oppure None se il file non esiste
    """
    if not os.path.exists(percorso):
        print(f"Errore : il file {percorso} non esiste.")
        return None
    return pd.read_csv(percorso)
