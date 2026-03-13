import unittest
from src.modelli import dividi_dataset
from src.valutazione import calcola_metriche
import pandas as pd

class TestDividiDataset(unittest.TestCase):

    def setUp(self):
        """Crea un DataFrame di test con 10 righe."""
        self.df = pd.DataFrame({
            'Feature1': range(10),
            'Feature2': range(10, 20),
            'Target': [0, 1] * 5  # 10 elementi in totale
        })

    def test_divisione_proporzioni(self):
        """Verifica che le dimensioni di train e test siano corrette (test_size=0.2)."""
        # Con 10 righe e test_size=0.2, ci aspettiamo 2 righe nel test e 8 nel train
        X_train, X_test, y_train, y_test = dividi_dataset(self.df, 'Target', test_size=0.2)

        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(y_test), 2)
        self.assertEqual(len(y_train), 8)

    def test_separazione_target(self):
        """Verifica che la colonna target sia stata rimossa da X e isolata in y."""
        X_train, X_test, y_train, y_test = dividi_dataset(self.df, 'Target')

        # 'Target' non deve essere nelle colonne di X
        self.assertNotIn('Target', X_train.columns)
        self.assertNotIn('Target', X_test.columns)
        # X deve contenere ancora le altre feature
        self.assertIn('Feature1', X_train.columns)

    def test_errore_test_size_invalido(self):
        """Verifica che venga sollevato ValueError per test_size fuori range."""
        with self.assertRaises(ValueError):
            dividi_dataset(self.df, 'Target', test_size=1.5)
        with self.assertRaises(ValueError):
            dividi_dataset(self.df, 'Target', test_size=-0.1)

    def test_errore_target_inesistente(self):
        """Verifica che venga sollevato KeyError se la colonna target manca."""
        with self.assertRaises(KeyError):
            dividi_dataset(self.df, 'Colonna_Sbagliata')

    def test_riproducibilita_random_state(self):
        """Verifica che lo stesso random_state produca la stessa divisione."""
        res1 = dividi_dataset(self.df, 'Target', random_state=42)
        res2 = dividi_dataset(self.df, 'Target', random_state=42)

        # Verifichiamo che gli indici del test set siano identici
        pd.testing.assert_index_equal(res1[1].index, res2[1].index)


class TestCalcolaMetriche(unittest.TestCase):

    def setUp(self):
        """Prepara due array di test: uno reale e uno predetto."""
        # Usiamo Series di Pandas per simulare l'output tipico dei modelli
        self.y_true = pd.Series([100.0, 200.0, 300.0])
        self.y_pred = pd.Series([110.0, 190.0, 300.0])

        # Calcoli manuali per la verifica:
        # Errori: [+10, -10, 0]
        # Assoluti: [10, 10, 0] -> MAE = 20/3 = 6.666...
        # Quadrati: [100, 100, 0] -> MSE = 200/3 = 66.666...
        # MAPE: [(10/100), (10/200), (0/300)] -> (0.1 + 0.05 + 0) / 3 * 100 = 5.0%

    def test_valori_metriche_base(self):
        """Verifica che MAE, MSE e RMSE siano calcolati correttamente."""
        risultati = calcola_metriche(self.y_true, self.y_pred)

        self.assertAlmostEqual(risultati['MAE'], 6.6666667, places=5)
        self.assertAlmostEqual(risultati['MSE'], 66.6666667, places=5)
        self.assertAlmostEqual(risultati['RMSE'], 66.6666667 ** 0.5, places=5)

    def test_r2_perfetto(self):
        """Verifica che il punteggio R2 sia 1.0 se le predizioni sono identiche ai valori reali."""
        risultati = calcola_metriche(self.y_true, self.y_true)
        self.assertEqual(risultati['R2 Score'], 1.0)

    def test_mape_calcolo(self):
        """Verifica il calcolo della percentuale di errore (MAPE)."""
        risultati = calcola_metriche(self.y_true, self.y_pred)
        # In base ai nostri dati nel setUp, il MAPE atteso è 5.0%
        self.assertAlmostEqual(risultati['MAPE'], 5.0, places=2)

    def test_output_formato(self):
        """Verifica che la funzione restituisca tutte le chiavi richieste nel dizionario."""
        risultati = calcola_metriche(self.y_true, self.y_pred)
        chiavi_attese = {'MAE', 'MSE', 'RMSE', 'R2 Score', 'MAPE'}
        self.assertTrue(chiavi_attese.issubset(risultati.keys()))


if __name__ == '__main__':
    unittest.main()