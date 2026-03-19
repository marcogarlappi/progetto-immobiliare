"""
Progetto Previsione Prezzi Immobiliari
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

Script principale che orchestra l' intero pipeline di analisi.

Utilizzo:
python main.py --fase tutte
python main.py --fase caricamento
python main.py --fase analisi
python main.py --fase modelli
python main.py --help

Autore: Marco Garlappi
Data: 06/03/2026
"""

import sys

import pandas as pd

# Importazione dei moduli del progetto
from src.data_loader import carica_dataset, salva_csv
from src.data_cleaning import (
    info_dataset, gestisci_valori_nulli,
    normalizza_colonne
    )
from src.analisi_esplorativa import (
    statistiche_descrittive, matrice_correlazione,
    genera_report_testuale
    )
from src.modelli import addestra_tutti_i_modelli, dividi_dataset
from src.valutazione import (
    calcola_metriche, cross_validation_modello,
    confronta_modelli, genera_report_modelli
    )
from src.utils import calcola_tempo_esecuzione, timestamp_corrente


def mostra_aiuto():
    """ Mostra le istruzioni d'uso del programma. """
    print("Utilizzo:")
    print("python main.py --fase tutte")
    print("python main.py --fase caricamento")
    print("python main.py --fase analisi")
    print("python main.py --fase modelli")
    print("python main.py --help")

def fase_caricamento():
    """ Fase 1: Caricamento e salvataggio dei dati. """
    data, target = carica_dataset()
    df_completo = pd.concat([data, target], axis=1)
    salva_csv(df_completo, "data/dataset_salvato.csv")
    return df_completo

def fase_pulizia(df):
    """ Fase 2: Pulizia e preprocessing dei dati. """
    df = gestisci_valori_nulli(df, strategia="media")
    df = normalizza_colonne(df, df.columns[:-1]) # Normalizziamo tutte le colonne tranne l'ultima (target)
    return df

def fase_analisi(df):
    """ Fase 3: Analisi esplorativa dei dati. """
    genera_report_testuale(df, "output/report.txt")

def fase_modelli(df):
    """ Fase 4: Addestramento e valutazione dei modelli. """
    X_train, X_test, y_train, y_test = dividi_dataset(df, df.columns[-1], test_size=0.2, random_state=42)
    risultati = addestra_tutti_i_modelli(X_train, y_train, X_test)
    genera_report_modelli(risultati, y_test, "output/report_modelli.txt")

def main():
    """ Funzione principale che gestisce il flusso del programma. """
    print(f"{'=' * 55}")
    print(f"PROGETTO PREVISIONE PREZZI IMMOBILIARI")
    print(f"Avviato il: {timestamp_corrente()}")
    print(f"{'=' * 55}\n")

    # Gestione argomenti da riga di comando

    if len(sys.argv) < 2 or '--help' in sys.argv:
        mostra_aiuto ()
        return

    # Parsing degli argomenti
    try:
        indice_fase = sys.argv.index('--fase')
        fase = sys.argv[indice_fase + 1]
    except (ValueError, IndexError):
        print("Errore: specificare --fase seguito dal nome della fase")
        mostra_aiuto ()
        return

    # Esecuzione della fase richiesta
    fasi_disponibili = {
        'caricamento': fase_caricamento,
        'analisi': fase_analisi,
        'modelli': fase_modelli,
        'tutte': None # Gestito separatamente
    }

    if fase not in fasi_disponibili :
        print(f"Errore: fase '{fase}' non riconosciuta.")
        print (f"Fasi disponibili: {', '.join(fasi_disponibili.keys())}")
        return

    if fase == 'tutte':
        df = fase_caricamento()
        df = fase_pulizia(df)
        fase_analisi(df)
        fase_modelli(df)
    else:
        # Implementare la logica per eseguire una singola fase
        pass

    print(f"\n{'=' * 55}")
    print(f"ESECUZIONE COMPLETATA")
    print(f"Terminato il: {timestamp_corrente()}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()