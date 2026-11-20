import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_model import model_artifact_paths, normalize_ticker
from stock_utils import (
    build_feature_matrix,
    build_test_prediction_frame,
    calculate_metrics,
    normalize_stock_data,
    prepare_partition,
    require_prediction_history,
)


class NormalizeStockDataTests(unittest.TestCase):
    def test_flattens_yfinance_multiindex_and_drops_nan_rows(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Close", "RELIANCE.BO"),
                ("Open", "RELIANCE.BO"),
                ("High", "RELIANCE.BO"),
            ]
        )
        data = pd.DataFrame(
            [[100.0, 98.0, 101.0], [np.nan, 99.0, 102.0]],
            columns=columns,
        )

        result = normalize_stock_data(data, "RELIANCE.BO")

        self.assertEqual(list(result.columns), ["Close", "Open", "High"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Close"].iloc[0], 100.0)

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "No historical data"):
            require_prediction_history(pd.DataFrame(), "INVALID.NS")

    def test_rejects_fewer_rows_than_lookback(self):
        data = pd.DataFrame({"Close": range(59)})

        with self.assertRaisesRegex(ValueError, "at least 60"):
            require_prediction_history(data, "RELIANCE.BO")

    def test_accepts_exact_lookback(self):
        data = pd.DataFrame({"Close": range(60)})

        self.assertIs(require_prediction_history(data, "RELIANCE.NS"), data)


class TrainingScalerTests(unittest.TestCase):
    def test_training_scaler_is_fit_only_on_training_rows(self):
        close = np.arange(100, 220, dtype=float)
        data = pd.DataFrame(
            {"Close": close, "High": close + 2, "Low": close - 2}
        )
        X_train, y_train, scaler, train_indices = prepare_partition(
            data,
            "close_only",
            fit_end=80,
            target_start=60,
            target_end=80,
        )
        X_test, _, _, test_indices = prepare_partition(
            data,
            "close_only",
            fit_end=80,
            target_start=80,
            target_end=len(data),
            scaler=scaler,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scaler.pkl"
            joblib.dump(scaler, path)
            loaded_scaler = joblib.load(path)

        self.assertEqual(scaler.data_max_[0], close[79])
        self.assertEqual(scaler.data_min_[0], close[0])
        self.assertEqual(X_train.shape, (20, 60, 1))
        self.assertEqual(y_train.shape, (20,))
        self.assertEqual(train_indices[0], 60)
        self.assertEqual(X_test.shape, (40, 60, 1))
        self.assertEqual(test_indices[0], 80)
        self.assertEqual(loaded_scaler.data_max_[0], close[79])

    def test_artifacts_are_specific_to_nse_and_bse_tickers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nse_model, nse_scaler, nse_metrics = model_artifact_paths(
                "reliance.ns", temp_dir
            )
            bse_model, bse_scaler, bse_metrics = model_artifact_paths(
                "RELIANCE.BO", temp_dir
            )

        self.assertEqual(nse_model.name, "RELIANCE.NS.keras")
        self.assertEqual(nse_scaler.name, "RELIANCE.NS.scaler.pkl")
        self.assertEqual(nse_metrics.name, "RELIANCE.NS.metrics.json")
        self.assertEqual(bse_model.name, "RELIANCE.BO.keras")
        self.assertEqual(bse_scaler.name, "RELIANCE.BO.scaler.pkl")
        self.assertEqual(bse_metrics.name, "RELIANCE.BO.metrics.json")
        self.assertNotEqual(nse_model, bse_model)

    def test_rejects_ticker_without_supported_exchange_suffix(self):
        with self.assertRaisesRegex(ValueError, "ending in .NS"):
            normalize_ticker("RELIANCE")


class FeatureTests(unittest.TestCase):
    def test_52_week_distances_are_real_features(self):
        close = np.arange(100, 140, dtype=float)
        data = pd.DataFrame(
            {
                "Close": close,
                "High": close + 5,
                "Low": close - 5,
            }
        )

        features = build_feature_matrix(data, "52_week_distances")

        self.assertEqual(features.shape, (40, 3))
        self.assertAlmostEqual(features[0, 1], 100 / 105 - 1)
        self.assertAlmostEqual(features[0, 2], 100 / 95 - 1)
        self.assertAlmostEqual(features[0, 0], 100)


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_price_units_and_root_mean_squared_error(self):
        metrics = calculate_metrics([10, 20, 30], [12, 16, 30])

        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], (20 / 3) ** 0.5)

    def test_metrics_reject_non_finite_predictions(self):
        with self.assertRaisesRegex(ValueError, "must all be finite"):
            calculate_metrics([100, 110], [np.nan, 109])

    def test_test_chart_frame_is_rebuilt_from_market_data_dates(self):
        class FakeModel:
            def __call__(self, features, training=False):
                return np.full((len(features), 1), 0.5)

        close = np.arange(100, 200, dtype=float)
        dates = pd.date_range("2020-01-01", periods=len(close), tz="Asia/Kolkata")
        data = pd.DataFrame(
            {"Close": close, "High": close + 2, "Low": close - 2},
            index=dates,
        )
        _, _, scaler, _ = prepare_partition(
            data,
            "close_only",
            fit_end=80,
            target_start=60,
            target_end=80,
        )

        frame = build_test_prediction_frame(
            FakeModel(),
            data,
            scaler,
            "close_only",
            str(dates[80]),
            str(dates[84]),
        )

        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.index[0], dates[80])
        self.assertEqual(frame["Actual"].iloc[0], close[80])
        self.assertEqual(frame["Naive baseline"].iloc[0], close[79])


if __name__ == "__main__":
    unittest.main()
