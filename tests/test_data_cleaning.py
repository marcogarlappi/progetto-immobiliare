import unittest
import pandas as pd

from src.data_cleaning import (
    info_dataset, gestisci_valori_nulli,
    rileva_outlier, normalizza_colonne,
)


class TestInfoDataset(unittest.TestCase):

    def setUp(self):
        """ Crea un DataFrame di test ."""

        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10.0, 20.0, None, 40.0, 50.0],
            'C': [100, 200, 300, 400, 500]
        })

        self.risultato = info_dataset(self.df)

    def test_numero_righe(self):
        """ Verifica che il numero di righe sia corretto ."""

        self.assertEqual(self.risultato['numero_righe'], 5)

    def test_numero_colonne(self):
        """ Verifica che il numero di colonne sia corretto ."""

        self.assertEqual(self.risultato['numero_colonne'], 3)

    def test_valori_nulli_rilevati(self):
        """ Verifica che i valori nulli vengano rilevati ."""

        self.assertEqual(self.risultato['valori_nulli']['B'], 1)
        self.assertEqual(self.risultato['valori_nulli']['A'], 0)
        self.assertEqual(self.risultato['valori_nulli']['C'], 0)
        self.assertEqual(self.risultato['percentuale_nulli']['B'], 20.0)

    def test_tipi_dati(self):
        """ Verifica che i tipi di dati siano identificati correttamente ."""

        self.assertTrue(pd.api.types.is_integer_dtype(self.risultato['tipi_dati']['A']))
        self.assertTrue(pd.api.types.is_float_dtype(self.risultato['tipi_dati']['B']))


class TestGestisciValoriNulli(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1.0, 2.0, None, 4.0, 5.0],
            'B': [10.0, None, 30.0, None, 50.0]
        })

    def test_strategia_media(self):
        """ Verifica che la strategia ’ media ’ funzioni . """
        df_pulito = gestisci_valori_nulli(self.df, strategia="media")
        self.assertEqual(df_pulito.isnull().sum().sum(), 0)
        self.assertEqual(df_pulito.loc[2, 'A'], 3.0)


    def test_strategia_mediana(self):
        """ Verifica che la strategia ’ mediana ’ funzioni . """
        df_pulito = gestisci_valori_nulli(self.df, strategia="mediana")
        self.assertEqual(df_pulito.isnull().sum().sum(), 0)
        self.assertEqual(df_pulito.loc[2, 'A'], 3.0)


    def test_strategia_elimina(self):
        """ Verifica che la strategia ’ elimina ’ rimuova le righe . """
        df_pulito = gestisci_valori_nulli(self.df, strategia="elimina")
        self.assertEqual(len(df_pulito), 2)
        self.assertEqual(df_pulito.isnull().sum().sum(), 0)


    def test_strategia_invalida(self):
        """ Verifica che una strategia non valida sollevi ValueError . """
        with self.assertRaises(ValueError):
            gestisci_valori_nulli(self.df, strategia="invalida")


    def test_nessun_nullo(self):
        """ Verifica il comportamento con DataFrame senza nulli . """
        df_pieno = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_risultato = gestisci_valori_nulli(df_pieno, strategia="media")
        pd.testing.assert_frame_equal(df_pieno, df_risultato)


class TestRilevaOutlier(unittest.TestCase):

    def test_outlier_rilevati(self):
        """ Verifica che gli outlier vengano rilevati con IQR. """
        df = pd.DataFrame({'prezzi': [1, 2, 3, 4, 5, 100]})
        indici_outlier = rileva_outlier(df, 'prezzi', metodo="iqr")
        self.assertIn(5, indici_outlier)
        self.assertEqual(len(indici_outlier), 1)

    def test_colonna_inesistente(self):
        """ Verifica che una colonna inesistente sollevi un errore. """
        df = pd.DataFrame({'A': [1, 2, 3]})
        with self.assertRaises(KeyError):
            rileva_outlier(df, 'colonna_fantasma', metodo="iqr")



class TestNormalizzaColonne(unittest.TestCase):

    def setUp(self):
        """Crea un DataFrame di test."""
        self.df = pd.DataFrame({
            'A': [10, 20, 30, 40, 50],
            'B': [100, 200, 300, 400, 500]
        })

    def test_minmax_range(self):
        """ Verifica che dopo la normalizzazione minmax i valori siano tra 0 e 1. """
        df_norm = normalizza_colonne(self.df, ['A', 'B'], metodo="minmax")
        for col in ['A', 'B']:
            self.assertEqual(df_norm[col].min(), 0.0)
            self.assertEqual(df_norm[col].max(), 1.0)
            self.assertTrue((df_norm[col] >= 0).all() and (df_norm[col] <= 1).all())

    def test_standard_media_zero(self):
        """ Verifica che dopo la standa rdizzazione la media sia ~0. """
        df_std = normalizza_colonne(self.df, ['A'], metodo="standard")

        media_ottenuta = df_std['A'].mean()
        dev_std_ottenuta = df_std['A'].std()

        self.assertAlmostEqual(media_ottenuta, 0.0, places=7)
        self.assertAlmostEqual(dev_std_ottenuta, 1.0, places=7)



if __name__ == '__main__':
    unittest.main()