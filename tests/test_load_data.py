from pathlib import Path
import unittest

import pandas as pd

from trade_analysis_script import load_data


class LoadDataTests(unittest.TestCase):
    DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data"
    USECOLS = [
        "TransDateTime",
        "Symbol",
        "Quantity",
        "BuySell",
        "FillPrice",
        "OpenClose",
        "HighDuringPosition",
        "LowDuringPosition",
    ]

    def test_load_data_with_naive_timestamps(self):
        df = load_data(self.DATA_DIR / "trade_log_sample.txt", usecols=self.USECOLS)

        self.assertIsNotNone(df["TransDateTime"].dt.tz)
        self.assertEqual(df["TransDateTime"].dt.tz.zone, "US/Eastern")

        expected_first = pd.Timestamp("2023-08-14 05:35:00-04:00", tz="US/Eastern")
        self.assertEqual(df.loc[0, "TransDateTime"], expected_first)

    def test_load_data_with_utc_timestamps(self):
        df = load_data(
            self.DATA_DIR / "trade_log_sample_utc.txt", usecols=self.USECOLS
        )

        self.assertIsNotNone(df["TransDateTime"].dt.tz)
        self.assertEqual(df["TransDateTime"].dt.tz.zone, "US/Eastern")

        expected_first = pd.Timestamp("2023-08-14 05:35:00-04:00", tz="US/Eastern")
        self.assertEqual(df.loc[0, "TransDateTime"], expected_first)


if __name__ == "__main__":  # pragma: no cover - convenience for direct execution
    unittest.main()
