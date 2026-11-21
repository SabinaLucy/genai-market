"""Regression tests for forecast horizon indexing (no model training)."""
import unittest
import numpy as np
import pandas as pd
from src.modeling import VIXDataset

class HorizonAlignmentTests(unittest.TestCase):
    def test_one_five_ten_days_after_last_input(self):
        frame = pd.DataFrame({"feature": np.arange(100, dtype=float),
                              "vix_log": np.arange(100, dtype=float)})
        dataset = VIXDataset(frame, ["feature"], sequence_length=60, horizons=[1, 5, 10])
        self.assertEqual(len(dataset), 31)
        x, y, idx = dataset[0]
        self.assertEqual(idx, 60)
        self.assertEqual(float(x[-1, 0]), 59.0)
        np.testing.assert_array_equal(y.numpy(), [60.0, 64.0, 69.0])
        _, last_y, last_idx = dataset[len(dataset) - 1]
        self.assertEqual(last_idx, 90)
        np.testing.assert_array_equal(last_y.numpy(), [90.0, 94.0, 99.0])

if __name__ == "__main__":
    unittest.main()
