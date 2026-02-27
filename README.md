# **NVIDIA Stock Forecasting — ML & Quant Research Pipeline**

This repository implements a **modular, production‑grade forecasting pipeline** for NVIDIA (NVDA) stock returns.  
It follows a clean quant‑research architecture with:

- reproducible data ingestion  
- feature engineering  
- target construction  
- model training (linear, tree‑based, ARIMA, LSTM-ready)  
- evaluation  
- experiment scripts  

The project is designed to be **extensible**, **config‑driven**, and **easy to audit**, following the structure used in real quant research teams.

---

## **📁 Project Structure**

```
├─ data/
│  ├─ raw/                # Raw downloaded NVDA data
│  ├─ processed/          # Cleaned data, features, targets
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py           # Central config paths & model registry
│  ├─ data_pipeline.py    # Load + clean + compute returns
│  ├─ features.py         # Feature engineering (lags, RSI, MACD, etc.)
│  ├─ targets.py          # Target generation (next return, direction)
│  ├─ train.py            # Model training logic
│  ├─ evaluate.py         # Evaluation utilities
│  ├─ plots.py            # Plotting utilities (predictions, curves)
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ linear_models.py # Linear, Ridge, Lasso, ElasticNet
│  │  ├─ tree_models.py   # RandomForest, GradientBoosting
│  │  ├─ lstm_model.py    # LSTM model (optional)
│  │  ├─ arima_models.py  # ARIMA baseline
│
├─ scripts/
│  ├─ run_download_data.py   # Download NVDA data
│  ├─ run_feature_build.py   # Build features + targets
│  ├─ run_train_all.py       # Train all models
│  ├─ run_evaluate_all.py    # Evaluate all models
│
├─ requirements.txt
├─ README.md
```

---

## **🚀 Quick Start**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

---

## **2. Download NVDA data**

```
python -m scripts.run_download_data
```

This saves:

```
data/raw/NVDA.csv
```

---

## **3. Build features + targets**

```
python -m scripts.run_feature_build
```

This generates:

- `data/processed/nvda.csv`  
- `data/processed/nvda_features.csv`  
- `data/processed/nvda_targets.csv`  

---

## **4. Train all models**

```
python -m scripts.run_train_all
```

Models are saved to:

```
src/models/*.pkl
```

---

## **5. Evaluate all models**

```
python -m scripts.run_evaluate_all
```

Results are written to:

```
results/eval_results.csv
```

---

## **📊 Models Included**

| Model Type | Models |
|-----------|--------|
| Linear Models | Linear, Ridge, Lasso, ElasticNet |
| Tree-Based | RandomForest, GradientBoosting |
| Time-Series | ARIMA |
| Deep Learning | LSTM (optional, modular) |

All models follow a unified interface defined in `src/train.py`.

---

## **🧠 Features Engineered**

The feature pipeline includes:

- **Lagged returns** (1, 5, 10, 20 days)  
- **Rolling means & volatility**  
- **RSI (14)**  
- **MACD (12/26/9)**  
- **Stochastic oscillator**  
- **Log returns**  
- **Daily returns**

All features are built in `src/features.py`.

---

## **🎯 Targets Generated**

Targets include:

- **Next‑day return**
- **5‑day cumulative return**
- **Direction (up/down)**

Defined in `src/targets.py`.

---

## **📈 Evaluation Metrics**

The evaluation pipeline computes:

- **MAE** (default)
- (Extendable to RMSE, MAPE, R², directional accuracy)

Results are saved to:

```
results/eval_results.csv
```

---

## **🧩 Extending the Project**

You can easily add:

- new models → `src/models/`
- new features → `src/features.py`
- new targets → `src/targets.py`
- new evaluation metrics → `src/evaluate.py`

The entire system is config‑driven via:

```
src/config.py
```

---

## **📌 Why This Project Matters**

This repository demonstrates:

- modular quant‑research engineering  
- reproducible ML forecasting workflows  
- clean separation of data, features, targets, models, and scripts  
- hedge‑fund‑style pipeline design  
- practical forecasting on real financial data  
