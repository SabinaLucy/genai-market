"""Test point-in-time FRED alignment without live credentials."""
import unittest
import pandas as pd
from scripts.repair_macro_history import vintage_asof

class MacroVintageTests(unittest.TestCase):
    def test_not_visible_before_release_and_revisions(self):
        releases = pd.DataFrame([
            {"date": "2026-01-01", "realtime_start": "2026-02-10", "value": 100},
            {"date": "2026-02-01", "realtime_start": "2026-03-12", "value": 110},
            {"date": "2026-01-01", "realtime_start": "2026-03-15", "value": 105},
        ])
        dates = pd.to_datetime(["2026-02-09", "2026-02-10", "2026-03-11",
                                "2026-03-12", "2026-03-15"])
        result = vintage_asof(releases, dates)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(result.iloc[1], 100)
        self.assertEqual(result.iloc[2], 100)
        self.assertEqual(result.iloc[3], 110)
        self.assertEqual(result.iloc[4], 110)

if __name__ == "__main__":
    unittest.main()
