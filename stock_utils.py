import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


LOOKBACK = 60
_OHLCV_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "adj close",
    "volume",
    "dividends",
    "stock splits",
}


def normalize_stock_data(data, ticker=None):
    """Normalize Yahoo Finance columns and remove rows with invalid values."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Stock data must be a pandas DataFrame.")
    if data.empty:
        return data.copy()

    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        columns = []
        for column in normalized.columns:
            parts = [str(part) for part in column]
            fields = [part for part in parts if part.casefold() in _OHLCV_FIELDS]
            if fields:
                field = fields[0]
            elif ticker and ticker.casefold() in {part.casefold() for part in parts}:
                field = next(part for part in parts if part.casefold() == ticker.casefold())
            else:
                field = parts[0]
            columns.append(field)
        normalized.columns = columns

    close_columns = [
        column for column in normalized.columns
        if str(column).casefold() == "close"
    ]
    if len(close_columns) != 1:
        raise ValueError("Yahoo Finance data must contain exactly one Close column.")
    if close_columns[0] != "Close":
        normalized = normalized.rename(columns={close_columns[0]: "Close"})

    normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return normalized


def require_prediction_history(data, ticker, lookback=LOOKBACK):
    """Return clean data or raise a user-readable validation error."""
    if data.empty:
        raise ValueError(
            f"No historical data was found for {ticker}. Check the ticker symbol "
            "and try again."
        )
    if len(data) < lookback:
        raise ValueError(
            f"Only {len(data)} valid daily rows were found for {ticker}; "
            f"at least {lookback} are required to make a prediction."
        )
    return data


def fetch_stock_data(ticker, period="5y"):
    data = yf.Ticker(ticker).history(period=period, timeout=20)
    return normalize_stock_data(data, ticker)


def build_feature_matrix(data, feature_mode):
    close = data["Close"].to_numpy(dtype=float)
    if feature_mode == "close_only":
        return close.reshape(-1, 1)
    if feature_mode != "52_week_distances":
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    if "High" not in data or "Low" not in data:
        raise ValueError("High and Low prices are required for 52-week features.")

    rolling_high = (
        pd.to_numeric(data["High"], errors="coerce")
        .rolling(window=252, min_periods=1)
        .max()
        .to_numpy(dtype=float)
    )
    rolling_low = (
        pd.to_numeric(data["Low"], errors="coerce")
        .rolling(window=252, min_periods=1)
        .min()
        .to_numpy(dtype=float)
    )
    return np.column_stack(
        (close, close / rolling_high - 1.0, close / rolling_low - 1.0)
    )


def prepare_partition(
    data,
    feature_mode,
    fit_end,
    target_start,
    target_end,
    scaler=None,
):
    """Scale on the training prefix and build ordered lookback sequences."""
    features = build_feature_matrix(data, feature_mode)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features[:fit_end])
    scaled = scaler.transform(features)

    target_indices = np.arange(max(LOOKBACK, target_start), target_end)
    if target_indices.size == 0:
        raise ValueError("Not enough rows to build sequences for this time split.")
    X = np.stack(
        [scaled[index - LOOKBACK:index] for index in target_indices]
    ).astype(np.float32)
    y = scaled[target_indices, 0].astype(np.float32)
    return X, y, scaler, target_indices


def inverse_scale_close(scaled_close, scaler):
    values = np.asarray(scaled_close, dtype=float).reshape(-1)
    scaled_features = np.zeros((len(values), scaler.n_features_in_), dtype=float)
    scaled_features[:, 0] = values
    return scaler.inverse_transform(scaled_features)[:, 0]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted price arrays must have matching values.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted prices must all be finite.")
    errors = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def predict_next_close(model, data, scaler, feature_mode):
    require_prediction_history(data, "the selected ticker")
    features = build_feature_matrix(data, feature_mode)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled[-LOOKBACK:][np.newaxis, ...], verbose=0)
    close = float(inverse_scale_close(prediction, scaler)[0])
    if not np.isfinite(close):
        raise ValueError("The model returned an invalid prediction.")
    return close


def build_test_prediction_frame(
    model,
    data,
    scaler,
    feature_mode,
    start_date,
    end_date,
):
    start_index = data.index.get_indexer([pd.Timestamp(start_date)])[0]
    end_index = data.index.get_indexer([pd.Timestamp(end_date)])[0]
    if start_index < 0 or end_index < start_index:
        raise ValueError(
            "The saved test dates are no longer available in Yahoo Finance data. "
            "Retrain this ticker to refresh its evaluation."
        )

    X_test, _, _, indices = prepare_partition(
        data,
        feature_mode,
        fit_end=start_index,
        target_start=start_index,
        target_end=end_index + 1,
        scaler=scaler,
    )
    scaled_predictions = np.asarray(model(X_test, training=False)).reshape(-1)
    predicted = inverse_scale_close(scaled_predictions, scaler)
    close_prices = data["Close"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "Actual": close_prices[indices],
            "LSTM": predicted,
            "Naive baseline": close_prices[indices - 1],
        },
        index=data.index[indices],
    )
