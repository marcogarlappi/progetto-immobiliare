"""
In main.py implementare la lettura di argomenti da riga di comando usando sys.argv:
• python main.py –fase tutte → esegue tutto il pipeline
• python main.py –fase caricamento → esegue solo il caricamento
• python main.py –fase analisi → esegue solo l’analisi
• python main.py –fase modelli → esegue solo l’addestramento dei modelli
• python main.py –help → mostra le istruzioni d’uso

"""
import sys
from src.data_loader import carica_dataset, salva_csv, carica_csv

def main():
    if len(sys.argv) < 2:
        print("Errore: nessuna fase specificata. Usa -help per le istruzioni.")
        return

    fase = sys.argv[1]

    if fase == "-fase":
        if len(sys.argv) < 3:
            print("Errore: specifica una fase (tutte, caricamento, analisi, modelli).")
            return
        subfase = sys.argv[2]
        if subfase == "tutte":
            print("Eseguo tutto il pipeline...")
            # chiamare tutte le funzioni per eseguire il pipeline completo
        elif subfase == "caricamento":
            print("Eseguo solo il caricamento...")
            data, target = carica_dataset()
            df = data.copy()
            df['target'] = target
            salva_csv(df, 'data/california_housing.csv')
        elif subfase == "analisi":
            dataframe = carica_csv('data/california_housing.csv')
            print("Eseguo solo l'analisi...")
            if dataframe is not None:
                print(dataframe.head())
                print(dataframe.describe())
            # chiamare le funzioni per eseguire l'analisi dei dati
        elif subfase == "modelli":
            print("Eseguo solo l'addestramento dei modelli...")
            # chiamare le funzioni per eseguire l'addestramento dei modelli
        else:
            print("Errore: fase non riconosciuta. Usa -help per le istruzioni.")
    elif fase == "-help":
        print("Istruzioni d'uso:")
        print("python main.py -fase tutte -> esegue tutto il pipeline")
        print("python main.py -fase caricamento -> esegue solo il caricamento del dataset")
        print("python main.py -fase analisi -> esegue solo l'analisi")
        print("python main.py -fase modelli -> esegue solo l'addestramento dei modelli")
    else:
        print("Errore: argomento non riconosciuto. Usa -help per le istruzioni.")

if __name__ == "__main__":
    main()