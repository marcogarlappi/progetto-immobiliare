import numpy as np
from scipy import stats
import os
import datetime
from src.data_cleaning import rileva_outlier


def statistiche_descrittive(df) -> dict:
    """
    Calcola statistiche descrittive per ogni colonna numerica :
    - media , mediana , moda
    - deviazione standard , varianza
    - minimo , massimo , range
    - primo quartile (Q1), terzo quartile (Q3), IQR
    - skewness , kurtosis

    Utilizza NumPy e SciPy per i calcoli .

    Returns :
    dict : dizionario annidato { nome_colonna : { statistica : valore }}
    """

    statistiche = {}
    for colonna in df.select_dtypes(include=[np.number]).columns:
        dati = df[colonna].dropna()
        mode_result = stats.mode(dati, keepdims=True)
        statistiche[colonna] = {
            'media': np.mean(dati),
            'mediana': np.median(dati),
            'moda': mode_result.mode[0],     #valore che appare più volte
            'deviazione_standard': np.std(dati, ddof=1),
            'varianza': np.var(dati, ddof=1),
            'minimo': np.min(dati),
            'massimo': np.max(dati),
            'range': np.ptp(dati),
            'Q1': np.percentile(dati, 25),
            'Q3': np.percentile(dati, 75),
            'IQR': stats.iqr(dati),
            'skewness': stats.skew(dati),       #asimmetria dei dati, sbilancio a destra o a sinistra rispetto alla media
            'kurtosis': stats.kurtosis(dati)    #appiattimento o picco dei dati rispetto alla distribuzione normale
        }
    return statistiche


def matrice_correlazione(df) -> "DataFrame":
    """
    Calcola e restituisce la matrice di correlazione.

    Identifica le coppie di feature con correlazione > 0.7 o < -0.7.

    Returns:
    DataFrame con la matrice di correlazione
    """

    correlazione = df.corr()
    coppie_correlate = []
    for i in range(len(correlazione.columns)):
        for j in range(i + 1, len(correlazione.columns)):
            if abs(correlazione.iloc[i, j]) > 0.7:
                coppie_correlate.append((correlazione.columns[i], correlazione.columns[j], correlazione.iloc[i, j]))
    print("Coppie di feature con correlazione > 0.7 o < -0.7:")
    for coppia in coppie_correlate:
        print(f"{coppia[0]} e {coppia[1]}: {coppia[2]:.2f}")
    return correlazione


def analisi_distribuzione(df, colonna: str) -> dict:
    """
    Analizza la distribuzione di una colonna specifica.

    Utilizza scipy.stats per:
    - Test di normalita (Shapiro-Wilk se possibile, o altri)
    - Calcolo skewness e kurtosis

    Returns:
    dict con i risultati dell’analisi
    """

    dati = df[colonna].dropna()
    risultato = {
        'shapiro_statistic': None,
        'shapiro_pvalue': None,
        'skewness': stats.skew(dati),
        'kurtosis': stats.kurtosis(dati)
    }
    try:
        shapiro_statistic, shapiro_pvalue = stats.shapiro(dati)
        risultato['shapiro_statistic'] = shapiro_statistic
        risultato['shapiro_pvalue'] = shapiro_pvalue
    except Exception as e:
        print(f"Shapiro-Wilk test non eseguibile: {e}")

    return risultato


def genera_report_testuale(df ,percorso_output: str) -> None:
    """
    Genera un report completo in formato testo (.txt) con:
    - Riepilogo del dataset
    - Statistiche descrittive formattate
    - Correlazioni significative
    - Osservazioni sugli outlier

    Utilizza f-string e string.format() per la formattazione.
    Salva il report nella cartella output/.

    Esempio di report:
        =======================================================
                    REPORT ANALISI IMMOBILIARE
                    Data: 2025 -01 -15 14:30:22
        =======================================================

        RIEPILOGO DATASET
        -----------------------------------------------------
        Numero di campioni: 20 ,640
        Numero di feature: 8
        Feature target: MedHouseVal

        STATISTICHE DESCRITTIVE
        -----------------------------------------------------
        Colonna Media Mediana StdDev Min Max
        MedInc 3.8707 3.5348 1.8998 0.4999
        15.0001
        HouseAge 28.6395 29.0000 12.5856 1.0000
        52.0000
        ...
    """

    # Assicuriamoci che la cartella output/ esista
    os.makedirs(os.path.dirname(percorso_output), exist_ok=True)

    # Raccogliamo i dati dalle funzioni precedenti
    stats_dict = statistiche_descrittive(df)
    corr_matrix = df.corr()  # Chiamata interna per semplicità nel report
    data_ora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(percorso_output, "w", encoding="utf-8") as f:
        # 1. INTESTAZIONE
        f.write("=" * 65 + "\n")
        f.write(f"{'REPORT ANALISI DATASET':^65}\n")
        f.write(f"{'Data: ' + data_ora:^65}\n")
        f.write("=" * 65 + "\n\n")

        # 2. RIEPILOGO DATASET
        f.write("RIEPILOGO DATASET\n")
        f.write("-" * 55 + "\n")
        f.write(f"Numero di campioni: {len(df):,}\n")
        f.write(f"Numero di feature:  {len(df.columns)}\n")
        f.write(f"Colonne presenti:   {', '.join(df.columns[:5])}...\n\n")

        # 3. STATISTICHE DESCRITTIVE (Tabellare)
        f.write("STATISTICHE DESCRITTIVE\n")
        f.write("-" * 55 + "\n")
        # Header tabella: {:<12} allinea a sinistra, {:>10} allinea a destra
        header = "{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Colonna", "Media", "Mediana", "StdDev", "Min", "Max"
        )
        f.write(header + "\n")

        for col, s in stats_dict.items():
            riga = "{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                col[:14], s['media'], s['mediana'], s['deviazione_standard'], s['minimo'], s['massimo']
            )
            f.write(riga + "\n")
        f.write("\n")

        # 4. OSSERVAZIONI OUTLIER
        f.write("ANALISI OUTLIER (Metodo IQR)\n")
        f.write("-" * 55 + "\n")
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers_indices = rileva_outlier(df, col, metodo="iqr")
            n_outliers = len(outliers_indices)
            percentuale = (n_outliers / len(df)) * 100
            f.write(f"- {col:<15}: {n_outliers:>6} outlier rilevati ({percentuale:>5.2f}%)\n")
        f.write("\n")

        # 5. CORRELAZIONI SIGNIFICATIVE
        f.write("CORRELAZIONI SIGNIFICATIVE (|r| > 0.7)\n")
        f.write("-" * 55 + "\n")
        trovate = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    f.write(f"- {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {r:.4f}\n")
                    trovate = True
        if not trovate:
            f.write("Nessuna correlazione forte rilevata.\n")

    print(f"Report generato con successo in: {percorso_output}")