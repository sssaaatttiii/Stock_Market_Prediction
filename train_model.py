import argparse
import json
from pathlib import Path

import joblib
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential

from stock_utils import (
    LOOKBACK,
    calculate_metrics,
    fetch_stock_data,
    inverse_scale_close,
    prepare_partition,
    require_prediction_history,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
FEATURE_MODES = ("close_only", "52_week_distances")


def normalize_ticker(ticker):
    ticker = str(ticker).strip().upper()
    if not ticker.endswith((".NS", ".BO")) or not ticker[:-3]:
        raise ValueError("Enter a ticker ending in .NS (NSE) or .BO (BSE).")
    if any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&._-" for char in ticker):
        raise ValueError("Ticker contains unsupported characters.")
    return ticker


def model_artifact_paths(ticker, models_dir=MODELS_DIR):
    ticker = normalize_ticker(ticker)
    directory = Path(models_dir)
    return (
        directory / f"{ticker}.keras",
        directory / f"{ticker}.scaler.pkl",
        directory / f"{ticker}.metrics.json",
    )


class _ProgressCallback(Callback):
    def __init__(self, progress_callback, epochs, progress_start, progress_span):
        super().__init__()
        self.progress_callback = progress_callback
        self.epochs = epochs
        self.progress_start = progress_start
        self.progress_span = progress_span

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback is not None:
            fraction = self.progress_start + self.progress_span * (
                (epoch + 1) / self.epochs
            )
            self.progress_callback(fraction, epoch + 1, self.epochs)


def build_lstm(input_features=1):
    model = Sequential(
        [
            Input(shape=(LOOKBACK, input_features)),
            LSTM(50, return_sequences=True),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    progress_callback=None,
    progress_start=0.0,
    progress_span=1.0,
):
    model = build_lstm(input_features=X_train.shape[2])
    callbacks = []
    if progress_callback is not None:
        callbacks.append(
            _ProgressCallback(
                progress_callback,
                epochs,
                progress_start,
                progress_span,
            )
        )
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    return model


def _predict_partition(model, X, scaler):
    scaled_predictions = model(X, training=False).numpy().reshape(-1)
    return inverse_scale_close(scaled_predictions, scaler)


def train_and_save_model(stock_name, epochs=20, progress_callback=None):
    ticker = normalize_ticker(stock_name)
    print(f"Fetching five years of data for {ticker}...")
    data = fetch_stock_data(ticker, period="5y")
    require_prediction_history(data, ticker, LOOKBACK + 2)

    row_count = len(data)
    test_start = int(row_count * 0.80)
    validation_start = int(test_start * 0.75)
    if validation_start <= LOOKBACK or test_start <= validation_start:
        raise ValueError("Not enough history for chronological train/validation/test splits.")

    validation_scores = {}
    candidate_progress_span = 0.25
    for candidate_index, feature_mode in enumerate(FEATURE_MODES):
        X_train, y_train, scaler, _ = prepare_partition(
            data,
            feature_mode,
            fit_end=validation_start,
            target_start=LOOKBACK,
            target_end=validation_start,
        )
        X_validation, _, _, validation_indices = prepare_partition(
            data,
            feature_mode,
            fit_end=validation_start,
            target_start=validation_start,
            target_end=test_start,
            scaler=scaler,
        )
        model = train_model(
            X_train,
            y_train,
            epochs=epochs,
            progress_callback=progress_callback,
            progress_start=candidate_index * candidate_progress_span,
            progress_span=candidate_progress_span,
        )
        validation_predicted = _predict_partition(model, X_validation, scaler)
        validation_actual = data["Close"].to_numpy(dtype=float)[validation_indices]
        validation_scores[feature_mode] = calculate_metrics(
            validation_actual,
            validation_predicted,
        )
        print(
            f"{feature_mode} validation MAE: "
            f"{validation_scores[feature_mode]['mae']:.4f}"
        )

    selected_mode = min(
        FEATURE_MODES,
        key=lambda mode: validation_scores[mode]["mae"],
    )
    print(f"Selected input features: {selected_mode}")

    X_train, y_train, scaler, _ = prepare_partition(
        data,
        selected_mode,
        fit_end=test_start,
        target_start=LOOKBACK,
        target_end=test_start,
    )
    model = train_model(
        X_train,
        y_train,
        epochs=epochs,
        progress_callback=progress_callback,
        progress_start=2 * candidate_progress_span,
        progress_span=0.5,
    )

    X_test, _, _, test_indices = prepare_partition(
        data,
        selected_mode,
        fit_end=test_start,
        target_start=test_start,
        target_end=row_count,
        scaler=scaler,
    )
    test_predicted = _predict_partition(model, X_test, scaler)
    close_prices = data["Close"].to_numpy(dtype=float)
    test_actual = close_prices[test_indices]
    baseline_predicted = close_prices[test_indices - 1]
    lstm_metrics = calculate_metrics(test_actual, test_predicted)
    baseline_metrics = calculate_metrics(test_actual, baseline_predicted)
    test_results = {
        "lstm": lstm_metrics,
        "naive_baseline": baseline_metrics,
        "model_beats_baseline": (
            lstm_metrics["mae"] < baseline_metrics["mae"]
            and lstm_metrics["rmse"] < baseline_metrics["rmse"]
        ),
    }

    model_path, scaler_path, metrics_path = model_artifact_paths(ticker)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    metadata = {
        "ticker": ticker,
        "feature_mode": selected_mode,
        "lookback": LOOKBACK,
        "target": "next_close",
        "validation_mae_by_feature_mode": validation_scores,
        "test": {
            **test_results,
            "start_date": str(data.index[test_indices[0]]),
            "end_date": str(data.index[test_indices[-1]]),
            "rows": int(len(test_indices)),
        },
    }
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Test LSTM MAE/RMSE: "
        f"{lstm_metrics['mae']:.4f} / {lstm_metrics['rmse']:.4f}"
    )
    print(
        f"Test naive MAE/RMSE: "
        f"{baseline_metrics['mae']:.4f} / {baseline_metrics['rmse']:.4f}"
    )
    print(
        "The LSTM beats the naive baseline on both metrics."
        if test_results["model_beats_baseline"]
        else "The LSTM does not beat the naive baseline on both metrics."
    )
    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved evaluation: {metrics_path}")
    if progress_callback is not None:
        progress_callback(1.0, epochs, epochs)
    return model_path, scaler_path, metrics_path, metadata


def main():
    parser = argparse.ArgumentParser(description="Train a stock-price LSTM.")
    parser.add_argument("ticker", nargs="?", default="RELIANCE.NS")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_and_save_model(args.ticker, epochs=args.epochs)


if __name__ == "__main__":
    main()
