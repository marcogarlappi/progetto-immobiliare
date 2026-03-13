import unittest
import math

from src.utils import (
    formatta_numero, formatta_percentuale,
    arrotonda_intelligente, calcola_distanza_euclidea,
    genera_campione_casuale,
)


class TestFormattaNumero(unittest.TestCase):

    def test_formattazione_standard(self):
        """ Verifica la formattazione con 2 decimali (default). """
        # Il separatore delle migliaia nel formato f"{:,.2f}" è la virgola
        risultato = formatta_numero(1234.567)
        self.assertEqual(risultato, "1,234.57") # Nota l'arrotondamento del 7

    def test_separatore_migliaia(self):
        """ Verifica che i numeri grandi abbiano le virgole correttamente posizionate. """
        risultato = formatta_numero(1000000)
        self.assertEqual(risultato, "1,000,000.00")

    def test_decimali_personalizzati(self):
        """ Verifica la formattazione con un numero specifico di decimali. """
        numero = 3.14159
        self.assertEqual(formatta_numero(numero, 0), "3")
        self.assertEqual(formatta_numero(numero, 4), "3.1416") # Arrotonda l'ultima cifra

    def test_numeri_negativi(self):
        """ Verifica che funzioni correttamente con i numeri negativi. """
        risultato = formatta_numero(-1500.5)
        self.assertEqual(risultato, "-1,500.50")



class TestFormattaPercentuale(unittest.TestCase):

    def test_formattazione_standard(self):
        """ Verifica la conversione standard con 1 decimale (default). """
        # 0.1234 deve diventare 12.3% (arrotondando alla prima cifra)
        risultato = formatta_percentuale(0.1234)
        self.assertEqual(risultato, "12.3%")

    def test_decimali_personalizzati(self):
        """ Verifica che il numero di decimali sia rispettato. """
        valore = 0.85678
        # Con 0 decimali
        self.assertEqual(formatta_percentuale(valore, 0), "86%") # Arrotonda 85.6 a 86
        # Con 3 decimali
        self.assertEqual(formatta_percentuale(valore, 3), "85.678%")

    def test_valori_maggiori_di_uno(self):
        """ Verifica il comportamento con valori superiori a 1 (es. crescita). """
        # 1.5 deve diventare 150.0%
        self.assertEqual(formatta_percentuale(1.5), "150.0%")

    def test_valore_zero(self):
        """ Verifica la formattazione dello zero. """
        self.assertEqual(formatta_percentuale(0), "0.0%")



class TestArrotondaIntelligente(unittest.TestCase):

    def test_valore_grande(self):
        """Verifica l'arrotondamento all'intero per valori > 1000."""
        self.assertEqual(arrotonda_intelligente(1234.567), 1235.0)
        self.assertEqual(arrotonda_intelligente(1000.1), 1000.0)

    def test_valore_medio(self):
        """Verifica l'arrotondamento a 2 decimali per valori tra 1 e 1000."""
        self.assertEqual(arrotonda_intelligente(123.4567), 123.46)
        self.assertEqual(arrotonda_intelligente(1.555), 1.55)

    def test_valore_piccolo(self):
        """Verifica l'arrotondamento a 4 decimali per valori tra 0.01 e 1."""
        self.assertEqual(arrotonda_intelligente(0.123456), 0.1235)
        self.assertEqual(arrotonda_intelligente(0.010101), 0.0101)

    def test_valore_microscopico(self):
        """Verifica la notazione scientifica per valori <= 0.01."""
        # 0.00012345 -> 1.23e-04
        self.assertEqual(arrotonda_intelligente(0.00012345), 1.23e-04)
        # Nota: il test confronta float, quindi 0.000123 è lo stesso di 1.23e-4
        self.assertEqual(arrotonda_intelligente(0.005), 5.0e-03)

    def test_valori_negativi(self):
        """Verifica che la logica funzioni (o come si comporta) con i negativi."""
        # Dato che usi 'valore > 1000', i negativi finiranno sempre nel ramo else (notazione scientifica)
        # È importante saperlo per evitare sorprese!
        self.assertEqual(arrotonda_intelligente(-500), -5.00e+02)



class TestCalcolaDistanzaEuclidea(unittest.TestCase):

    def test_distanza_2d_semplice(self):
        """ Verifica il calcolo base in 2D (triangolo pitagorico 3-4-5). """
        # Punti: (0,0) e (3,4) -> Distanza = sqrt(3^2 + 4^2) = 5
        p1 = [0, 0]
        p2 = [3, 4]
        self.assertEqual(calcola_distanza_euclidea(p1, p2), 5.0)

    def test_punti_identici(self):
        """ La distanza tra un punto e se stesso deve essere 0. """
        p1 = [1.5, 2.5, 3.5]
        p2 = [1.5, 2.5, 3.5]
        self.assertEqual(calcola_distanza_euclidea(p1, p2), 0.0)

    def test_distanza_3d(self):
        """ Verifica il calcolo in 3 dimensioni. """
        p1 = [1, 1, 1]
        p2 = [2, 2, 2]
        # Distanza = sqrt((2-1)^2 + (2-1)^2 + (2-1)^2) = sqrt(3)
        risultato_atteso = math.sqrt(3)
        self.assertAlmostEqual(calcola_distanza_euclidea(p1, p2), risultato_atteso, places=7)

    def test_dimensioni_diverse(self):
        """ Verifica che venga sollevato ValueError se i punti hanno dimensioni diverse. """
        p1 = [1, 2]
        p2 = [1, 2, 3] # Tre dimensioni invece di due
        with self.assertRaises(ValueError):
            calcola_distanza_euclidea(p1, p2)

    def test_coordinate_negative(self):
        """ Verifica che funzioni correttamente con coordinate negative. """
        p1 = [-1, -1]
        p2 = [1, 1]
        # sqrt((1 - (-1))^2 + (1 - (-1))^2) = sqrt(2^2 + 2^2) = sqrt(8)
        self.assertAlmostEqual(calcola_distanza_euclidea(p1, p2), math.sqrt(8), places=7)


class TestGeneraCampioneCasuale(unittest.TestCase):

    def setUp(self):
        """Crea una lista di test con 100 elementi."""
        self.lista_test = list(range(100))

    def test_dimensione_campione(self):
        """Verifica che la dimensione del campione sia corretta (10% di 100 = 10)."""
        campione = genera_campione_casuale(self.lista_test, percentuale=0.1)
        self.assertEqual(len(campione), 10)

    def test_riproducibilita_seed(self):
        """Verifica che lo stesso seed generi lo stesso identico campione."""
        campione1 = genera_campione_casuale(self.lista_test, seed=42)
        campione2 = genera_campione_casuale(self.lista_test, seed=42)
        self.assertEqual(campione1, campione2)

    def test_seed_diversi(self):
        """Verifica che seed diversi generino campioni diversi."""
        campione1 = genera_campione_casuale(self.lista_test, seed=42)
        campione2 = genera_campione_casuale(self.lista_test, seed=7)
        self.assertNotEqual(campione1, campione2)

    def test_campione_minimo(self):
        """Verifica che venga estratto almeno 1 elemento anche con percentuali basse."""
        lista_piccola = [1, 2, 3]
        # 10% di 3 sarebbe 0.3, ma il max(1, ...) deve garantire 1 elemento
        campione = genera_campione_casuale(lista_piccola, percentuale=0.01)
        self.assertEqual(len(campione), 1)
        self.assertTrue(campione[0] in lista_piccola)

    def test_elementi_unici(self):
        """Verifica che random.sample non estragga lo stesso elemento più volte."""
        campione = genera_campione_casuale(self.lista_test, percentuale=0.5)
        # In un set i duplicati verrebbero rimossi; se la lunghezza resta uguale, sono unici
        self.assertEqual(len(campione), len(set(campione)))



if __name__ == '__main__':
    unittest.main()