import math
import random
from datetime import datetime, timedelta


def formatta_numero(numero: float, decimali: int = 2) -> str:
    """
    Formatta un numero con separatore delle migliaia e decimali specificati .

    Args :
    numero : numero da formattare
    decimali : numero di decimali da visualizzare ( default 2)

    Returns :
    str : numero formattato come stringa
    """

    formato = f"{{:,.{decimali}f}}"
    return formato.format(numero)


def formatta_percentuale(valore: float, decimali: int = 1) -> str:
    """
    Converte un valore decimale in stringa percentuale .

    Args :
    valore : numero decimale da convertire ( es. 0.1234)
    decimali : numero di decimali da visualizzare ( default 1)

    Returns :
    str : valore formattato come percentuale ( es. "12.3%")
    """

    formato = f"{{:.{decimali}%}}"
    return formato.format(valore)


def genera_colori_casuali(n: int, seed: int = 42) -> list:
    """
    Genera n colori casuali in formato esadecimale .
    Usa random . seed () per la riproducibilita .
    Utile per eventuali grafici o report .

    Args :
    n : numero di colori da generare
    seed : seed per la generazione casuale ( default 42)

    Returns :
    list : lista di stringhe con i colori esadecimali ( es. ["#1a2b3c", "#4d5e6f"])
    """

    random.seed(seed)
    colori = []
    for _ in range(n):
        colore = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colori.append(colore)
    return colori


def calcola_tempo_esecuzione(func):
    """
    Decoratore che misura e stampa il tempo di esecuzione
    di una funzione . Usa datetime per il calcolo .

    Args :
    func : funzione da decorare
    """

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        tempo_esecuzione = end_time - start_time
        print(f"Tempo di esecuzione di {func.__name__}: {tempo_esecuzione}")
        return result

    return wrapper


def crea_separatore(carattere: str = "=", lunghezza: int = 55) -> str:
    """
    Crea una stringa separatore per il report .

    Args :
    carattere : carattere da ripetere ( default "=")
    lunghezza : numero di caratteri da ripetere ( default 55)

    Returns :
    str : stringa separatore ( es. "=======================================================")
    """

    return carattere * lunghezza


def timestamp_corrente(formato: str = "%Y -%m -%d %H:%M:%S") -> str:
    """
    Restituisce il timestamp corrente formattato.

    Args :
    formato : formato del timestamp ( default "%Y-%m-%d %H:%M:%S")

    Returns :
    str : timestamp formattato ( es. "2025-01-15 14:30:22")
    """

    return datetime.now().strftime(formato)


def arrotonda_intelligente(valore: float) -> float:
    """
    Arrotonda un valore in modo intelligente :
    - Se > 1000: arrotonda all ’intero
    - Se > 1: arrotonda a 2 decimali
    - Se > 0.01: arrotonda a 4 decimali
    - Altrimenti : notazione scientifica

    Usa math .floor , math .ceil , math . log10

    Args :
    valore : numero da arrotondare

    Returns :
    float : numero arrotondato
    """

    if valore > 1000:
        return round(valore)
    elif valore > 1:
        return round(valore, 2)
    elif valore > 0.01:
        return round(valore, 4)
    else:
        return float(f"{valore:.2e}")


def genera_campione_casuale(lista: list, percentuale: float = 0.1,
                            seed: int = 42) -> list:
    """
    Estrae un campione casuale da una lista .
    Usa random.sample().

    Args :
    lista : lista di elementi da cui estrarre il campione
    percentuale : percentuale di elementi da estrarre ( default 0.1)
    seed : seed per la generazione casuale ( default 42)

    Returns :
    list : campione estratto dalla lista
    """

    random.seed(seed)
    n_campione = max(1, int(len(lista) * percentuale))
    campione = random.sample(lista, n_campione)
    return campione


def calcola_distanza_euclidea(punto1: list, punto2: list) -> float:
    """
    Calcola la distanza euclidea tra due punti n-dimensionali .
    Usa math.sqrt() e math.pow().

    Raises:
    ValueError: se i punti hanno dimensioni diverse

    Args:
    punto1 : lista di coordinate del primo punto ( es. [1, 2, 3])
    punto2 : lista di coordinate del secondo punto ( es. [4, 5, 6])

    Returns:
    float : distanza euclidea tra i due punti
    """

    if len(punto1) != len(punto2):
        raise ValueError("I punti devono avere la stessa dimensione.")

    somma = 0
    for coord1, coord2 in zip(punto1, punto2):
        somma += math.pow(coord2 - coord1, 2)

    distanza = math.sqrt(somma)
    return distanza


# Funzione placeholder per future implementazioni
def esporta_in_json(dati, percorso: str) -> None:
    """ Placeholder per futura implementazione ."""
    pass  # Da implementare in futuro
