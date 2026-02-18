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

    import numpy as np
    from scipy import stats

    statistiche = {}
    for colonna in df.select_dtypes(include=[np.number]).columns:
        dati = df[colonna].dropna()
        statistiche[colonna] = {
            'media': np.mean(dati),
            'mediana': np.median(dati),
            'moda': stats.mode(dati)[0][0],
            'deviazione_standard': np.std(dati, ddof=1),
            'varianza': np.var(dati, ddof=1),
            'minimo': np.min(dati),
            'massimo': np.max(dati),
            'range': np.ptp(dati),
            'Q1': np.percentile(dati, 25),
            'Q3': np.percentile(dati, 75),
            'IQR': stats.iqr(dati),
            'skewness': stats.skew(dati),
            'kurtosis': stats.kurtosis(dati)
        }
    return statistiche


def matrice_correlazione(df) -> "DataFrame":
    """
    Calcola e restituisce la matrice di correlazione .

    Identifica le coppie di feature con correlazione > 0.7 o < -0.7.

    Returns :
    DataFrame con la matrice di correlazione
    """

    import pandas as pd

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
    Analizza la distribuzione di una colonna specifica .

    Utilizza scipy . stats per:
    - Test di normalita ( Shapiro - Wilk se possibile , o altri )
    - Calcolo skewness e kurtosis

    Returns :
    dict con i risultati dell ’analisi
    """

    from scipy import stats

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


def genera_report_testuale ( df , percorso_output : str) -> None :
    """
    Genera un report completo in formato testo (. txt) con:
    - Riepilogo del dataset
    - Statistiche descrittive formattate
    - Correlazioni significative
    - Osservazioni sugli outlier

    Utilizza f- string e string . format () per la formattazione .
    Salva il report nella cartella output /.

    Esempio di report:
        =======================================================
                    REPORT ANALISI IMMOBILIARE
                    Data : 2025 -01 -15 14:30:22
        =======================================================

        RIEPILOGO DATASET
        -----------------------------------------------------
        Numero di campioni : 20 ,640
        Numero di feature : 8
        Feature target : MedHouseVal

        STATISTICHE DESCRITTIVE
        -----------------------------------------------------
        Colonna Media Mediana Std Dev Min Max
        MedInc 3.8707 3.5348 1.8998 0.4999
        15.0001
        HouseAge 28.6395 29.0000 12.5856 1.0000
        52.0000
        ...
    """

    import os
    from datetime import datetime

    if not os.path.exists('output'):
        os.makedirs('output')

    report = []
    report.append("=" * 60)
    report.append("REPORT ANALISI IMMOBILIARE")
    report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("\nRIEPILOGO DATASET")
    report.append("-" * 60)
    report.append(f"Numero di campioni: {df.shape[0]}")
    report.append(f"Numero di feature: {df.shape[1] - 1}")
    report.append(f"Feature target: {df.columns[-1]}")

    report.append("\nSTATISTICHE DESCRITTIVE")
    report.append("-" * 60)
    statistiche = statistiche_descrittive(df)
    for colonna, stats in statistiche.items():
        report.append(f"{colonna}: Media={stats['media']:.4f}, Mediana={stats['mediana']:.4f}, "
                      f"Deviazione Std={stats['deviazione_standard']:.4f}, Min={stats['minimo']:.4f}, "
                      f"Max={stats['massimo']:.4f}")

    with open(percorso_output, 'w') as f:
        f.write("\n".join(report))