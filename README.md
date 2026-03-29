The main pipeline is waam_ml_analysis.py — it does everything end-to-end:

Loads both CSVs
Filters active welding, creates time windows
Handles data imbalance
Trains RF & XGBoost (classification + regression)
Generates all plots: ROC, SHAP, confusion matrices, feature importance, regression scatter, correlation heatmap, data overview
The other scripts are one-off helpers for regenerating individual figures:

Make sure the two CSV files (new_aligned_data_good.csv, new_aligned_data_bad.csv) are in the same folder as the script. Everything outputs to a separate output folder.
