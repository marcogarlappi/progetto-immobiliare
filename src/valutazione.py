from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import os
from datetime import datetime


def calcola_metriche(y_true, y_pred) -> dict:
    """
    Calcola le metriche di valutazione per la regressione :
    - MAE ( Mean Absolute Error )
    - MSE ( Mean Squared Error )
    - RMSE ( Root Mean Squared Error )
    - R2 Score
    - MAPE ( Mean Absolute Percentage Error )

    Returns :
    dict con tutte le metriche calcolate
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    mape = (abs((y_true - y_pred) / y_true).mean()) * 100

    metriche = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'MAPE': mape
    }
    return metriche


def cross_validation_modello(modello, X, y, cv: int = 5) -> dict:
    """
    Esegue la cross - validation su un modello .

    Returns :
    dict con: ’scores ’, ’media ’, ’ deviazione_standard ’
    """

    scores = cross_val_score(modello, X, y, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Convertiamo in MSE positivo
    media_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)

    risultati_cv = {
        'scores': mse_scores,
        'media': media_mse,
        'deviazione_standard': std_mse
    }
    return risultati_cv


def confronta_modelli(risultati: dict, y_test) -> str:
    """
    Confronta tutti i modelli e determina il migliore .

    Args :
    risultati : dizionario con i risultati di ogni modello

    Returns :
    str : nome del modello migliore
    """

    miglior_modello = None
    minor_errore = float('inf')

    print("\n--- Report Performance Modelli (MSE) ---")

    for nome_modello, info in risultati.items():
        # Otteniamo le predizioni salvate nel dizionario di ogni modello
        predizioni = info['predizioni']

        # Calcoliamo l'errore (MSE) confrontando predizioni e valori reali (y_test)
        mse = ((predizioni - y_test) ** 2).mean()

        print(f"Modello: {nome_modello:20} | MSE: {mse:.4f}")

        # Se questo errore è il più basso visto finora, aggiorniamo il vincitore
        if mse < minor_errore:
            minor_errore = mse
            miglior_modello = nome_modello

    print("----------------------------------------")
    print(f"IL MIGLIOR MODELLO È: {miglior_modello.upper()}")

    return miglior_modello


def genera_report_modelli(risultati: dict, percorso_output: str) -> None:
    """
    Genera un report testuale dettagliato con i risultati
    di tutti i modelli , inclusa la cross - validation .

    Il report deve includere :
    - Tabella comparativa di tutti i modelli
    - Dettagli per ogni modello
    - Raccomandazione del modello migliore
    - Data e ora della generazione

    Salva nella cartella output /.
    """
    # 1. Crea la cartella di output se non esiste
    os.makedirs(os.path.dirname(percorso_output), exist_ok=True)

    ora_generazione = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(percorso_output, "w", encoding="utf-8") as f:
        # Intestazione
        f.write("=" * 50 + "\n")
        f.write(f"REPORT PRESTAZIONI MODELLI DI REGRESSIONE\n")
        f.write(f"Data generazione: {ora_generazione}\n")
        f.write("=" * 50 + "\n\n")

        # Sezione 1: Tabella Comparativa (Logica di confronto)
        f.write("1. TABELLA COMPARATIVA\n")
        f.write(f"{'Modello':25} | {'Parametro Migliore':20}\n")
        f.write("-" * 50 + "\n")

        miglior_modello_nome = ""
        minor_errore_cv = float('inf')

        for nome, info in risultati.items():
            # Cerchiamo il parametro migliore (k, profondità, o kernel)
            param = info.get('miglior_k', info.get('miglior_profondita', info.get('miglior_kernel', 'N/A')))
            f.write(f"{nome:25} | {str(param):20}\n")

            # Logica per determinare la raccomandazione (basata su CV se disponibile)
            # Nota: usiamo l'ultimo MSE di cross-validation per decidere
            mse_cv = info.get('mse_validazione', info.get('risultati_cv', {}).get(str(param), float('inf')))
            if isinstance(mse_cv, float) and mse_cv < minor_errore_cv:
                minor_errore_cv = mse_cv
                miglior_modello_nome = nome

        f.write("\n" + "=" * 50 + "\n")

        # Sezione 2: Dettagli per ogni modello
        f.write("2. DETTAGLI MODELLI\n")
        for nome, info in risultati.items():
            f.write(f"\n>>> MODELLO: {nome.upper()}\n")
            for chiave, valore in info.items():
                if chiave != 'predizioni' and chiave != 'modello':  # Escludiamo i dati pesanti
                    f.write(f"   {chiave}: {valore}\n")

        # Sezione 3: Raccomandazione finale
        f.write("\n" + "=" * 50 + "\n")
        f.write("3. RACCOMANDAZIONE FINALE\n")
        f.write(f"Basandosi sui risultati della Cross-Validation,\n")
        f.write(f"il modello consigliato è: {miglior_modello_nome.upper()}\n")
        f.write("=" * 50 + "\n")

    print(f"Report generato con successo in: {percorso_output}")



