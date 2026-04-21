The main pipeline is waam_ml_pipeline_final.py — it does everything end-to-end:

Loads all six CSVs

Trains RF & XGBoost (regression)
Generates all plots

Make sure the six CSV files are in the same folder as the script. Everything outputs to a separate output folder.
