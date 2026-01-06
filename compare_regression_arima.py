# # compare_regression_arima_full.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# 1. Load Regression results
# -----------------------------
reg_metrics_path = "data/processed/regression_metrics.json"
reg_pred_path = "data/processed/regression_predictions_sample.csv"

with open(reg_metrics_path, "r", encoding="utf-8") as f:
    reg_metrics = json.load(f)

reg_pred_df = pd.read_csv(reg_pred_path, parse_dates=['datetime'])

# -----------------------------
# 2. Load ARIMA results
# -----------------------------
arima_summary_path = "data/processed/arima_pm25_summary.json"
arima_pred_path = "data/processed/arima_pm25_predictions.csv"

with open(arima_summary_path, "r", encoding="utf-8") as f:
    arima_summary = json.load(f)

arima_pred_df = pd.read_csv(arima_pred_path, parse_dates=['datetime'])

# -----------------------------
# 3. Print metrics comparison
# -----------------------------
print("=== Metrics Comparison ===")
print(f"Regression MAE: {reg_metrics['mae']:.3f}, RMSE: {reg_metrics['rmse']:.3f}")
print(f"ARIMA MAE:      {arima_summary['mae']:.3f}, RMSE: {arima_summary['rmse']:.3f}")

# -----------------------------
# 4. Plot PM2.5 full period
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(reg_pred_df['datetime'], reg_pred_df['y_true'], color='black')
plt.title("PM2.5 full period")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Plot PM2.5 zoom 1-2 months
# -----------------------------
zoom_start, zoom_end = 0, 24*60  # ví dụ 60 ngày, mỗi giờ 1 giá trị
plt.figure(figsize=(12,4))
plt.plot(reg_pred_df['datetime'].iloc[zoom_start:zoom_end],
         reg_pred_df['y_true'].iloc[zoom_start:zoom_end], color='black')
plt.title("PM2.5 zoom 1-2 months")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Plot ACF & PACF
# -----------------------------
plt.figure(figsize=(12,4))
plot_acf(reg_pred_df['y_true'].dropna(), lags=50, alpha=0.05)
plt.title("ACF of PM2.5")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(reg_pred_df['y_true'].dropna(), lags=50, alpha=0.05)
plt.title("PACF of PM2.5")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Forecast vs Actual (ARIMA sample)
# -----------------------------
sample_arima = arima_pred_df.iloc[:500]

plt.figure(figsize=(12,5))
plt.plot(sample_arima['datetime'], sample_arima['y_true'], label='Actual', color='black')
plt.plot(sample_arima['datetime'], sample_arima['y_pred'], label='ARIMA forecast', color='red')
plt.fill_between(sample_arima['datetime'], sample_arima['lower'], sample_arima['upper'],
                 alpha=0.2, color='red', label='95% CI')
plt.title("Forecast vs Actual (ARIMA, sample 500 points)")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Spike comparison (manual 1-3 days)
# -----------------------------
spike_start, spike_end = 1000, 1100  # chọn spike cụ thể
spike_reg = reg_pred_df.iloc[spike_start:spike_end]
spike_arima = arima_pred_df.iloc[spike_start:spike_end]

plt.figure(figsize=(12,5))
plt.plot(spike_reg['datetime'], spike_reg['y_true'], label='Actual', color='black')
plt.plot(spike_reg['datetime'], spike_reg['y_pred'], label='Regression', color='blue')
plt.plot(spike_arima['datetime'], spike_arima['y_pred'], label='ARIMA', color='red')
plt.fill_between(spike_arima['datetime'], spike_arima['lower'], spike_arima['upper'],
                 alpha=0.2, color='red', label='ARIMA 95% CI')
plt.title("Spike Comparison: Regression vs ARIMA")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.legend()
plt.tight_layout()
plt.show()
