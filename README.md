# Indian Stock Price Prediction App

This local Streamlit app displays historical Indian stock prices from Yahoo
Finance and estimates the next trading session's closing price with an LSTM
model. It accepts NSE tickers ending in `.NS` and BSE tickers ending in `.BO`.

The estimate is experimental and is not investment advice.

## What the project contains

- `stock_app.py` — Streamlit interface, charts, model loading, and predictions.
- `stock_utils.py` — shared Yahoo Finance data cleanup, feature generation,
  sequence building, scaling, metrics, and prediction helpers.
- `train_model.py` — command-line training for one ticker at a time.
- `models/` — ticker-specific `.keras` models, their fitted scaler, and saved
  chronological evaluation results. The app never substitutes another ticker's
  model.

The old root-level `stock_prediction_model.h5` is a legacy artifact and is not
used by the app.

## Setup on Windows

Python 3.12 is recommended. From the project folder, create and activate a
virtual environment, then install the pinned packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train a ticker

Train from the command line:

```powershell
python train_model.py RELIANCE.NS
python train_model.py TCS.BO
```

The optional `--epochs` argument changes the number of training epochs, for
example `python train_model.py RELIANCE.NS --epochs 30`. The app also offers a
**Train model for this ticker** button when no complete artifact set exists.

Each ticker creates three files under `models/`:

- `<TICKER>.keras` — the model for that exact ticker.
- `<TICKER>.scaler.pkl` — the scaler fitted on that model's training portion.
- `<TICKER>.metrics.json` — validation feature comparison, test summary metrics,
  and date boundaries. Test chart values are rebuilt from the cached Yahoo
  Finance data at runtime rather than stored in this file.

Training fetches up to five years of daily data. It compares closing-price-only
inputs with distances from rolling 52-week highs and lows using an earlier,
chronological validation window. The lower-validation-MAE feature set is kept.
The current `RELIANCE.NS` model uses closing prices only; its 52-week high and
low are shown in the app as display-only values.

In a controlled five-epoch ablation on the chronological test window for
`RELIANCE.NS`, close-only inputs scored MAE/RMSE of ₹42.45/₹51.20, while adding
52-week distance features scored ₹45.22/₹55.26. The extra features did not help,
so the app labels the 52-week range as display-only. The saved model selection
uses the earlier validation window, keeping the final test window out of
feature selection.

The target remains the next trading session's close, and the input lookback is
60 trading sessions. Training does not shuffle sequences. The scaler is fit on
the training portion only. The last 20% of dates are held out for testing and
compared with a naive baseline that predicts tomorrow's close as today's close.
The app shows MAE and RMSE for both; it warns when the LSTM does not beat the
baseline on both metrics.

## Run locally

```powershell
streamlit run stock_app.py
```

The app downloads and caches historical data, shows the latest data and
52-week range, plots recent closes with the next-session estimate, and plots
actual, LSTM, and naive-baseline prices over the saved test period.

## Known limitations

- Yahoo Finance data availability can differ by exchange and ticker. Prediction
  requires at least 60 clean daily rows; training needs more history for the
  chronological validation and test splits.
- This is a simple historical-price model. It does not account for news,
  company fundamentals, market events, transaction costs, or future corporate
  actions.
- Historical test performance does not guarantee future performance. Retrain
  each ticker when you want to refresh its saved model and evaluation.
- Windows TensorFlow runs on CPU; training may take a while.
