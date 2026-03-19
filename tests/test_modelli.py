import unittest
from src.modelli import dividi_dataset
from src.valutazione import calcola_metriche
import pandas as pd
import numpy as np

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
        self.y_true = pd.Series([100.0, 200.0, 300.0])
        self.y_pred = pd.Series([110.0, 190.0, 300.0])

        # Riassunto calcoli manuali corretti per sklearn:
        # MAE: (10 + 10 + 0) / 3 = 6.666...
        # MSE: (100 + 100 + 0) / 3 = 66.666...
        # MAPE: (0.1 + 0.05 + 0) / 3 = 0.05  <-- Nota: sklearn usa il decimale

    def test_valori_metriche_base(self):
        """Verifica che MAE, MSE e RMSE siano calcolati correttamente."""
        risultati = calcola_metriche(self.y_true, self.y_pred)

        self.assertAlmostEqual(risultati['MAE'], 6.6666667, places=5)
        self.assertAlmostEqual(risultati['MSE'], 66.6666667, places=5)
        self.assertAlmostEqual(risultati['RMSE'], np.sqrt(66.6666667), places=5)

    def test_r2_perfetto(self):
        """Verifica che il punteggio R2 sia 1.0 se le predizioni sono identiche."""
        risultati = calcola_metriche(self.y_true, self.y_true)
        # Usiamo la chiave 'R2' come definito nella funzione
        self.assertEqual(risultati['R2'], 1.0)

    def test_mape_calcolo(self):
        """Verifica il calcolo della percentuale di errore (MAPE) in formato decimale."""
        risultati = calcola_metriche(self.y_true, self.y_pred)
        # Ci aspettiamo 0.05 (che corrisponde al 5%)
        self.assertAlmostEqual(risultati['MAPE'], 0.05, places=5)

    def test_output_formato(self):
        """Verifica che la funzione restituisca esattamente le chiavi richieste."""
        risultati = calcola_metriche(self.y_true, self.y_pred)
        chiavi_attese = {'MAE', 'MSE', 'RMSE', 'R2', 'MAPE'}
        self.assertEqual(set(risultati.keys()), chiavi_attese)


if __name__ == '__main__':
    unittest.main()