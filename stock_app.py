import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from stock_utils import (
    build_test_prediction_frame,
    fetch_stock_data,
    predict_next_close,
    require_prediction_history,
)
from train_model import model_artifact_paths, normalize_ticker, train_and_save_model


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_stock_data(ticker, period):
    return fetch_stock_data(ticker, period=period)


@st.cache_resource
def load_ticker_artifacts(ticker, artifact_signature):
    model_path, scaler_path, metrics_path = model_artifact_paths(ticker)
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model, scaler, metadata


def build_historical_chart(data, predicted_price):
    history = data.tail(252)
    history_frame = pd.DataFrame(
        {
            "Date": history.index,
            "Close": history["Close"].to_numpy(dtype=float),
        }
    )
    forecast_frame = pd.DataFrame(
        {
            "Date": [data.index[-1] + pd.offsets.BDay(1)],
            "Close": [predicted_price],
        }
    )
    history_chart = (
        alt.Chart(history_frame)
        .mark_line(color="#4A90E2")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Close:Q", title="Closing price (INR)"),
            tooltip=["Date:T", alt.Tooltip("Close:Q", format=".2f")],
        )
    )
    prediction_marker = (
        alt.Chart(forecast_frame)
        .mark_point(color="#E45756", size=140, filled=True)
        .encode(
            x="Date:T",
            y="Close:Q",
            tooltip=["Date:T", alt.Tooltip("Close:Q", format=".2f")],
        )
    )
    return (history_chart + prediction_marker).interactive()


def main():
    st.title("Indian Stock Price Prediction App")
    st.write("Explore historical prices and an LSTM-based next-day estimate.")
    st.info("Educational project only; predictions are not investment advice.")

    st.sidebar.header("User Input")
    raw_ticker = st.sidebar.text_input(
        "NSE or BSE stock ticker",
        "RELIANCE.NS",
        help="Use the exchange suffix .NS for NSE or .BO for BSE.",
    )
    st.sidebar.caption(
        "Examples: NSE RELIANCE.NS or TCS.NS; "
        "BSE RELIANCE.BO or TCS.BO."
    )

    try:
        ticker = normalize_ticker(raw_ticker)
    except ValueError as error:
        st.sidebar.error(str(error))
        return

    model_path, scaler_path, metrics_path = model_artifact_paths(ticker)
    model_available = all(
        path.is_file() for path in (model_path, scaler_path, metrics_path)
    )
    predict_clicked = st.sidebar.button("Predict")

    if not model_available:
        st.info(f"No complete model is available for {ticker}.")
        if any(path.exists() for path in (model_path, scaler_path, metrics_path)):
            st.warning("The saved ticker artifacts are incomplete. Retrain this ticker.")
        if st.button("Train model for this ticker"):
            progress = st.progress(0.0, text=f"Training {ticker}...")

            def update_progress(fraction, epoch, total_epochs):
                progress.progress(
                    fraction,
                    text=f"Training {ticker}: epoch {epoch}/{total_epochs}",
                )

            try:
                train_and_save_model(ticker, progress_callback=update_progress)
                progress.progress(1.0, text="Training complete.")
                st.success(f"Model for {ticker} is ready. Click Predict to continue.")
            except Exception as error:
                st.error(f"Could not train a model for {ticker}: {error}")
        if predict_clicked:
            st.warning("Train this ticker's model before requesting a prediction.")
        return

    if not predict_clicked:
        return

    st.write(f"Fetching data for {ticker}...")
    try:
        data = cached_fetch_stock_data(ticker, period="5y")
        require_prediction_history(data, ticker)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Could not fetch data for {ticker}: {error}")
        return

    st.subheader("Recent Stock Data")
    st.dataframe(data.tail())

    try:
        artifact_signature = tuple(
            (path.name, path.stat().st_mtime_ns)
            for path in (model_path, scaler_path, metrics_path)
        )
        model, scaler, metadata = load_ticker_artifacts(
            ticker,
            artifact_signature,
        )
        predicted_price = predict_next_close(
            model,
            data,
            scaler,
            metadata["feature_mode"],
        )
        test_metrics = metadata["test"]
        test_frame = build_test_prediction_frame(
            model,
            data,
            scaler,
            metadata["feature_mode"],
            test_metrics["start_date"],
            test_metrics["end_date"],
        )
    except Exception as error:
        st.error(f"Could not load or use the model for {ticker}: {error}")
        return

    st.success(f"Estimated next closing price for {ticker}: ₹{predicted_price:.2f}")
    st.subheader("Recent Closing Prices and Next-Session Estimate")
    st.altair_chart(
        build_historical_chart(data, predicted_price),
        width="stretch",
    )

    if "High" in data and "Low" in data:
        week_high = float(data["High"].tail(252).max())
        week_low = float(data["Low"].tail(252).min())
        high_low_columns = st.columns(2)
        high_low_columns[0].metric("52-week high (display only)", f"₹{week_high:.2f}")
        high_low_columns[1].metric("52-week low (display only)", f"₹{week_low:.2f}")

    selected_features = metadata["feature_mode"]
    st.caption(
        "52-week high/low distances are model inputs."
        if selected_features == "52_week_distances"
        else "52-week high/low are display-only; the model uses closing prices."
    )
    feature_scores = metadata["validation_mae_by_feature_mode"]
    st.caption(
        "Earlier validation MAE: close-only "
        f"₹{feature_scores['close_only']['mae']:.2f}; 52-week distances "
        f"₹{feature_scores['52_week_distances']['mae']:.2f}. "
        f"Selected: {selected_features}."
    )

    test_metrics = metadata["test"]
    st.subheader("Chronological Test-Set Results")
    lstm_metrics = test_metrics["lstm"]
    baseline_metrics = test_metrics["naive_baseline"]
    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "LSTM MAE / RMSE",
        f"₹{lstm_metrics['mae']:.2f} / ₹{lstm_metrics['rmse']:.2f}",
    )
    metric_columns[1].metric(
        "Naive baseline MAE / RMSE",
        f"₹{baseline_metrics['mae']:.2f} / ₹{baseline_metrics['rmse']:.2f}",
    )
    if test_metrics["model_beats_baseline"]:
        st.success("The LSTM beats the naive baseline on both MAE and RMSE.")
    else:
        st.warning(
            "The LSTM does not beat the naive baseline on both metrics. "
            "Treat its estimate cautiously."
        )

    st.subheader("Predicted vs Actual Prices on the Test Period")
    st.line_chart(
        test_frame,
        x_label="Test date",
        y_label="Closing price (INR)",
        width="stretch",
    )


if __name__ == "__main__":
    main()
