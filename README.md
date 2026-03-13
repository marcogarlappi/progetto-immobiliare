# Progetto Previsione Prezzi Immobiliari

## Descrizione
Un progetto end-to-end di Data Science & Machine Learning, un’applicazione Python completa che:
1. Carica e pulisce un dataset immobiliare.
2. Esegue un’analisi esplorativa dei dati.
3. Addestra e confronta diversi modelli di Machine Learning per prevedere i prezzi delle case.
4. Produce un report testuale con i risultati.
5. Include test unitari per le funzioni principali.

Il dataset utilizzato è il California Housing Dataset, disponibile direttamente in scikit- learn.

## Struttura del Progetto
progetto - immobiliare /


| - - README . md # Descrizione del progetto

| - - requirements . txt # Dipendenze del progetto

| - - main . py # Script principale ( entry point )
|

| - - src /

    | | - - __init__ . py
    | | - - data_loader . py # Caricamento e salvataggio dati
    | | - - data_cleaning . py # Pulizia e preprocessing
    | | - - an a li si_e spl orat iva . py # Analisi e statistiche
    | | - - modelli . py # Addestramento modelli ML
    | | - - valutazione . py # Valutazione e confronto modelli
    | | - - utils . py # Funzioni di utilita
|
| - - tests /

    | | - - __init__ . py
    | | - - test_d ata_c le aning . py
    | | - - test_utils . py
    | | - - test_modelli . py
|
| - - data /

    | | - - ( file generati dal programma )
|
| - - output /

    | | - - ( report e risultati generati )
|
| - - docs /

    | - - ( documentazione generata con pydoc )

## Requisiti
- Python 3.8+
- scikit-learn
- pandas
- numpy
- scipy

## Installazione
pip install -r requirements.txt

## Utilizzo
python main.py --fase tutte

python main.py --help

## Modelli Implementati
- Regressione lineare
- Decision Tree
- K-Nearest Neighbors
- Support Vector regressor

## Risultati
Vedere report modelli nella cartella output/.

## Testing
python -m unittest discover

## Documentazione
[ Come consultare la documentazione generata ]

## Autore
Marco Garlappi, marcogarlappi@italymail.biz
