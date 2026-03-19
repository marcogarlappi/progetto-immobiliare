from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import numpy as np


def cross_validation_modello(modello, X, y, cv: int = 5) -> dict:
    """
    Esegue la cross-validation su un modello.

    Returns:
    dict con: ’scores’, ’media’, ’deviazione_standard’
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


def calcola_metriche(y_true, y_pred) -> dict:
    """Calcola le principali metriche di valutazione per la regressione."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }


def confronta_modelli(risultati: dict, y_test) -> str:
    """Determina il modello migliore basandosi sul punteggio R2 più alto."""
    miglior_score = -float('inf')
    miglior_nome = ""

    for nome, dati in risultati.items():
        metriche = calcola_metriche(y_test, dati['predizioni'])
        # Usiamo l'R2 come discriminante (più è vicino a 1, meglio è)
        if metriche['R2'] > miglior_score:
            miglior_score = metriche['R2']
            miglior_nome = nome

    return miglior_nome


def genera_report_modelli(risultati: dict, y_test, percorso_output: str) -> None:
    """Genera un file di testo con il confronto dettagliato e il verdetto finale."""
    migliore = confronta_modelli(risultati, y_test)

    with open(percorso_output, 'w', encoding='utf-8') as f:
        f.write("==============================================\n")
        f.write("   REPORT VALUTAZIONE MODELLI - CALIFORNIA    \n")
        f.write("==============================================\n\n")

        for nome, dati in risultati.items():
            m = calcola_metriche(y_test, dati['predizioni'])

            f.write(f"--- MODELLO: {nome} ---\n")
            # Se presente, riportiamo il parametro ottimale trovato
            if 'miglior_k' in dati: f.write(f"Parametro scelto: k={dati['miglior_k']}\n")
            if 'miglior_profondita' in dati: f.write(f"Parametro scelto: depth={dati['miglior_profondita']}\n")
            if 'miglior_kernel' in dati: f.write(f"Parametro scelto: kernel={dati['miglior_kernel']}\n")

            f.write(f"MAE:  {m['MAE']:.4f}\n")
            f.write(f"MSE:  {m['MSE']:.4f}\n")
            f.write(f"RMSE: {m['RMSE']:.4f}\n")
            f.write(f"R2:   {m['R2']:.4f}\n")
            f.write(f"MAPE: {m['MAPE']:.2%}\n")
            f.write("-" * 30 + "\n\n")

        f.write("==============================================\n")
        f.write(f" RACCOMANDAZIONE FINALE: {migliore.upper()} \n")
        f.write("==============================================\n")

    print(f"Report generato con successo in: {percorso_output}")

