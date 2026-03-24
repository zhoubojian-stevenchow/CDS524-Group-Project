# European Weather Temperature Forecasting

## XGBoost vs Bidirectional LSTM with Attention — A City-by-City Comparative Study

**Course:** CDS524 Machine Learning Application — Group Project  
**Institution:** Lingnan University  
**Date:** March 2026

---

## Overview

This project tackles **next-day temperature forecasting** (mean, min, max) for 5 European cities using the [European Cities Weather Prediction Dataset](https://www.kaggle.com/datasets/orvile/european-cities-weather-prediction-dataset) (ECA&D, 3,654 daily observations, 2000–2010). We compare **XGBoost** (gradient boosting on tabular features) against a **Bidirectional LSTM with Attention** (deep learning on 30-day sliding windows), and conduct a **2×2 factorial controlled variable experiment** to isolate the effects of lag/rolling features and Chinese cosmological features (24 Solar Terms 二十四节气, Wuxing 五行, Heavenly Stems 天干).

### Key Findings

- **XGBoost wins all 15/15 city-target pairs** in every configuration (avg MAE 1.78–1.82°C vs LSTM 2.51–2.71°C)
- **Chinese features help XGBoost only when no lag features exist** (A→B: −0.039°C, p=0.0005) but are **redundant** when lags are present (C→D: p=0.918)
- **Chinese features consistently harm LSTM** due to curse of dimensionality on ~2,900 training samples (C→D: +0.189°C, p=0.002)
- **Lag/rolling features provide surprisingly marginal gains** — same-day weather observations already contain most predictive signal for tomorrow's temperature

---

## Repository Structure

```
├── README.md
├── report/
│   └── Group Assignment -- European Weather Forecasting.docx
│
├── notebooks/
│   ├── Config_A_Raw_11_Only.ipynb                    # 11 raw features only
│   ├── Config_B_Raw_11_Plus_Chinese.ipynb             # 11 raw + Chinese features
│   ├── Config_C_Lag_Rolling_No_Chinese.ipynb          # 11 raw + lag/rolling (132 features)
│   └── Config_D_Full_Lag_Rolling_Plus_Chinese.ipynb   # 11 raw + lag/rolling + Chinese (150 features)
│
├── results/
│   ├── results_raw_11_only/                  # Config A outputs
│   │   ├── results_xgboost.csv
│   │   ├── results_lstm.csv
│   │   ├── results_comparison.csv
│   │   └── fig_01 ... fig_15 (.png)
│   ├── results_raw_11_plus_chinese/          # Config B outputs
│   │   └── ...
│   ├── results_baseline_no_chinese_202603101752/          # Config C outputs
│   │   └── ...
│   └── results_节气_五行_天干地支_202603101645/              # Config D outputs
│       ├── results_xgboost.csv
│       ├── results_lstm.csv
│       ├── results_comparison.csv
│       ├── xgboost_*.json                    # Saved XGBoost models
│       ├── lstm_*.pth                        # Saved LSTM models
│       └── fig_01 ... fig_15 (.png)
│
└── data/
    └── (download from Kaggle - see instructions below)
```

---

## Experiment Design

### 2×2 Factorial Controlled Variable Experiment

|                    | No Chinese Features | + Chinese Features |
|--------------------|:-------------------:|:------------------:|
| **Raw 11 Only**    | Config A (baseline) | Config B           |
| **+ Lag/Rolling**  | Config C            | Config D           |

- **5 cities:** DE_BILT, DUSSELDORF, MAASTRICHT, MUENCHEN, OSLO (all with complete 11-feature coverage)
- **3 targets per city:** next-day temp_mean, temp_min, temp_max
- **15 evaluation pairs** per model per configuration
- **Chronological split:** Train 2000–2007 | Val 2008 | Test 2009

### Models

| Model | Architecture | Tuning |
|-------|-------------|--------|
| **XGBoost** | Gradient boosting (hist method, GPU-accelerated) | Grid search: 36 hyperparameter combos per city-target (n_estimators, max_depth, learning_rate, min_child_weight) |
| **LSTM** | 2-layer Bidirectional LSTM (hidden=128) + Self-Attention + 3-layer MLP head | AdamW (lr=1e-3), ReduceLROnPlateau, early stopping (patience=20) |

### Feature Sets

| Config | XGBoost Features | LSTM Window Features | Description |
|--------|:----------------:|:--------------------:|-------------|
| A | 11 | 30-day × 11 | Raw weather only |
| B | 29 | 30-day × 29 | + 4 cyclical time + 4 solar term + 10 wuxing/stem |
| C | 132 | 30-day × 11 | + 77 lag (7-day) + 44 rolling (7/30-day mean/std) |
| D | 150 | 30-day × 29 | All features combined |

---

## Results Summary

### Average MAE (°C) Across All 15 City-Target Pairs

| Config | XGBoost MAE | LSTM MAE | XGBoost R² | LSTM R² |
|--------|:-----------:|:--------:|:----------:|:-------:|
| A: Raw 11 Only | 1.819 | **2.512** | 0.897 | **0.808** |
| B: Raw 11 + Chinese | **1.780** | 2.578 | **0.901** | 0.800 |
| C: Lag/Roll (no Chinese) | 1.790 | 2.519 | 0.898 | 0.805 |
| D: Lag/Roll + Chinese | 1.790 | 2.708 | 0.897 | 0.774 |

**Best overall:** XGBoost Config B (1.780°C MAE)  
**Best LSTM:** Config A with simplest features (2.512°C MAE)

### Statistical Significance (Paired t-Tests, n=15)

| Comparison | Δ MAE (°C) | p-value | Cohen's d |
|------------|:----------:|:-------:|:---------:|
| XGB: A→B (+ Chinese, no lags) | −0.039 | 0.0005*** | −1.20 |
| XGB: C→D (+ Chinese, with lags) | −0.001 | 0.918 ns | −0.03 |
| XGB: A→C (+ Lags, no Chinese) | −0.028 | 0.013* | −0.76 |
| LSTM: C→D (+ Chinese, with lags) | +0.189 | 0.002** | +1.01 |
| LSTM: B→D (+ Lags, with Chinese) | +0.130 | 0.0001*** | +1.52 |

---

## Setup and Reproduction

### Requirements

```
Python 3.10+
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
torch (PyTorch 2.x)
statsmodels
ephem                # for solar term computation (Configs B and D only)
```

### Step 1: Download Dataset

Download from Kaggle and place in your Google Drive:

```
My Drive/CDS524 Group Project/weather_prediction_dataset.csv
```

Dataset URL: https://www.kaggle.com/datasets/orvile/european-cities-weather-prediction-dataset

### Step 2: Run Notebooks

Each notebook is self-contained and runs on **Google Colab** (GPU recommended for LSTM training). Open in Colab and run all cells:

1. **Config A:** `notebooks/Config_A_Raw_11_Only.ipynb`
2. **Config B:** `notebooks/Config_B_Raw_11_Plus_Chinese.ipynb`
3. **Config C:** `notebooks/Config_C_Lag_Rolling_No_Chinese.ipynb`
4. **Config D:** `notebooks/Config_D_Full_Lag_Rolling_Plus_Chinese.ipynb`

Each notebook will:
- Mount Google Drive and load the dataset
- Run EDA (figures saved to results folder)
- Train XGBoost with grid search (36 combos × 3 targets × 5 cities)
- Train LSTM with early stopping (up to 200 epochs × 5 cities)
- Save results CSVs, model files, and figures to the corresponding results folder

**Estimated runtime per notebook:** ~30–45 min on Colab with T4 GPU.

### Step 3: View Results

Results CSVs contain MAE, RMSE, MAPE, and R² for all 15 city-target pairs. The `results_comparison.csv` in each results folder contains the head-to-head XGBoost vs LSTM comparison.

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean(\|y − ŷ\|) | Primary metric (°C) |
| RMSE | √mean((y − ŷ)²) | Penalises large errors |
| MAPE | mean(\|y − ŷ\| / \|y\|) × 100% | Unreliable for temperature (zero-crossing); reported with caveats |
| R² | 1 − SS_res / SS_tot | Explained variance |

**Note on MAPE:** Temperature values frequently cross 0°C in European winters, producing extreme percentage errors when actual values are near zero. XGBoost MAPE (37–115%) is inflated but reportable. LSTM MAPE exceeds 10⁶% for 9/15 city-target pairs and is marked N/A. See Hyndman & Koehler (2006) for discussion of this known limitation.

---

## References

1. Huber, F. et al. (2022). *European Cities Weather Prediction Dataset.* Zenodo/Kaggle.
2. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
3. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
4. Hyndman, R. J. & Koehler, A. B. (2006). *Another Look at Measures of Forecast Accuracy.* IJF, 22(4).
5. Klein Tank, A.M.G. et al. (2002). *Daily Dataset of 20th-Century Surface Air Temperature and Precipitation Series for the European Climate Assessment.* IJC, 22.
6. Bahdanau, D. et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.

---

## License

This project is for academic purposes (CDS524 coursework). The dataset is sourced from ECA&D via Kaggle under their respective licenses.
