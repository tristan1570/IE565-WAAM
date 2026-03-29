"""
WAAM Bead Geometry Prediction using Process Signals
====================================================
Analyzes current, voltage, acoustic amplitude, and IR thermography data
to predict bead geometry using Random Forest and XGBoost models.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"c:\Users\trist\OneDrive - University of Tennessee\Documents\new_data_test")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GOOD_CSV = BASE_DIR / "new_aligned_data_good.csv"
BAD_CSV = BASE_DIR / "new_aligned_data_bad.csv"

# Time window parameters
WINDOW_SIZE_SEC = 0.5    # 0.5 second windows
WINDOW_STEP_SEC = 0.25   # 0.25 second step (50% overlap)

# Feature groups
ELECTRICAL_FEATURES = ['Current(A)', 'Avg_Current(A)', 'Voltage(V)', 'Avg_Voltage(V)', 'Amplitude']
IR_FEATURES = ['mp_width_px', 'mp_height_px', 'mp_circularity', 'mp_aspect_ratio',
               'thermal_energy_haz', 'grad_mean_in_mp', 'global_p95', 'saturated_area_px']
TARGET_COLS = ['xiris_bead_width_mm', 'xiris_bead_height_mm', 'xiris_bead_area_mm2']

# Calibration from Xiris raw to actual mm (from CLAUDE.md)
CAL_HEIGHT = lambda x: 2.1718 * x - 26.6314
CAL_WIDTH = lambda x: 1.5822 * x - 13.7136
CAL_AREA = lambda x: 0.4061 * x - 30.8708

RANDOM_SEED = 42
N_FOLDS = 5

# ============================================================
# Data Loading & Preprocessing
# ============================================================

def load_and_filter_data(csv_path: str, label: str) -> pd.DataFrame:
    """Load CSV, filter to active welding period (where bead geometry > 0)."""
    logger.info(f"Loading {label} data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"  Raw shape: {df.shape}")

    # Filter to active welding: rows where bead geometry is detected
    active_mask = df['xiris_bead_width_mm'] > 0
    df_active = df[active_mask].copy()
    logger.info(f"  Active welding rows: {df_active.shape[0]}")

    # Drop rows with NaN in key features
    key_cols = ELECTRICAL_FEATURES + IR_FEATURES + TARGET_COLS
    available_cols = [c for c in key_cols if c in df_active.columns]
    df_active = df_active.dropna(subset=available_cols)
    logger.info(f"  After dropping NaN: {df_active.shape[0]}")

    # Add bead quality label
    df_active['bead_quality'] = label  # 'good' or 'bad'

    return df_active


def create_time_windows(df: pd.DataFrame, window_size: float, step_size: float) -> pd.DataFrame:
    """Create overlapping time windows and aggregate features."""
    t_min = df['rel_time'].min()
    t_max = df['rel_time'].max()

    windows = []
    t_start = t_min

    while t_start + window_size <= t_max:
        t_end = t_start + window_size
        mask = (df['rel_time'] >= t_start) & (df['rel_time'] < t_end)
        window_data = df[mask]

        if len(window_data) > 10:  # Minimum samples per window
            agg = {'window_start': t_start, 'window_end': t_end, 'n_samples': len(window_data)}

            # Aggregate features: mean and std for each
            for col in ELECTRICAL_FEATURES + IR_FEATURES:
                if col in window_data.columns:
                    agg[f'{col}_mean'] = window_data[col].mean()
                    agg[f'{col}_std'] = window_data[col].std()

            # Target: mean bead geometry in window
            for col in TARGET_COLS:
                agg[f'{col}_mean'] = window_data[col].mean()

            # Keep quality label
            agg['bead_quality'] = window_data['bead_quality'].iloc[0]

            windows.append(agg)

        t_start += step_size

    return pd.DataFrame(windows)


def apply_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Xiris-to-actual calibration for bead geometry."""
    df['bead_height_cal_mm'] = CAL_HEIGHT(df['xiris_bead_height_mm_mean'])
    df['bead_width_cal_mm'] = CAL_WIDTH(df['xiris_bead_width_mm_mean'])
    df['bead_area_cal_mm2'] = CAL_AREA(df['xiris_bead_area_mm2_mean'])
    return df


# ============================================================
# Feature Engineering
# ============================================================

def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Define feature groups for analysis."""
    elec_feats = []
    ir_feats = []
    for col in df.columns:
        for ef in ELECTRICAL_FEATURES:
            if col.startswith(ef) and ('_mean' in col or '_std' in col):
                elec_feats.append(col)
        for irf in IR_FEATURES:
            if col.startswith(irf) and ('_mean' in col or '_std' in col):
                ir_feats.append(col)

    return {
        'Electrical Only': sorted(set(elec_feats)),
        'IR Thermography Only': sorted(set(ir_feats)),
        'All Features Combined': sorted(set(elec_feats + ir_feats))
    }


# ============================================================
# Model Training & Evaluation
# ============================================================

def train_and_evaluate_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    group_name: str,
    n_folds: int = N_FOLDS
) -> Dict:
    """Train RF and XGBoost regressors with cross-validation."""
    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=5, random_state=RANDOM_SEED, n_jobs=-1,
            verbosity=0
        )
    }

    for model_name, model in models.items():
        # Cross-validation scores
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))

        # Cross-val predictions for plotting
        y_pred = cross_val_predict(model, X, y, cv=kf)

        # Fit final model on all data for feature importance
        model.fit(X, y)

        results[model_name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'y_true': y,
            'y_pred': y_pred,
            'model': model,
            'feature_names': feature_names
        }

        logger.info(
            f"  {model_name} | {group_name} | {target_name}: "
            f"R²={r2_scores.mean():.4f}±{r2_scores.std():.4f}, "
            f"MAE={mae_scores.mean():.4f}±{mae_scores.std():.4f}"
        )

    return results


def train_and_evaluate_classification(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    group_name: str,
    n_folds: int = N_FOLDS
) -> Dict:
    """Train RF and XGBoost classifiers for bead quality prediction."""
    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_child_weight=5, random_state=RANDOM_SEED, n_jobs=-1,
            verbosity=0, eval_metric='logloss'
        )
    }

    for model_name, model in models.items():
        acc_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')

        # Cross-val predictions
        y_pred = cross_val_predict(model, X, y, cv=kf)
        y_prob = cross_val_predict(model, X, y, cv=kf, method='predict_proba')[:, 1]

        # Final model
        model.fit(X, y)

        results[model_name] = {
            'acc_mean': acc_scores.mean(),
            'acc_std': acc_scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
            'y_true': y,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model': model,
            'feature_names': feature_names
        }

        logger.info(
            f"  {model_name} | {group_name}: "
            f"Acc={acc_scores.mean():.4f}±{acc_scores.std():.4f}, "
            f"F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}, "
            f"AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}"
        )

    return results


# ============================================================
# Visualization Functions
# ============================================================

def plot_data_overview(df_good_win: pd.DataFrame, df_bad_win: pd.DataFrame):
    """Plot overview of the windowed dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('WAAM Process Data Overview (Time-Windowed)', fontsize=16, fontweight='bold')

    # Current over time
    for ax_idx, (df, label, color) in enumerate([(df_good_win, 'Good Bead (#4)', 'tab:blue'),
                                                   (df_bad_win, 'Bad Bead (#2)', 'tab:red')]):
        axes[0, 0].plot(df['window_start'], df['Current(A)_mean'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_title('Current Over Time')
    axes[0, 0].legend()

    # Voltage over time
    for df, label, color in [(df_good_win, 'Good', 'tab:blue'), (df_bad_win, 'Bad', 'tab:red')]:
        axes[0, 1].plot(df['window_start'], df['Voltage(V)_mean'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].set_title('Voltage Over Time')
    axes[0, 1].legend()

    # Amplitude over time
    for df, label, color in [(df_good_win, 'Good', 'tab:blue'), (df_bad_win, 'Bad', 'tab:red')]:
        axes[0, 2].plot(df['window_start'], df['Amplitude_mean'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].set_title('Acoustic Amplitude Over Time')
    axes[0, 2].legend()

    # Bead width
    for df, label, color in [(df_good_win, 'Good', 'tab:blue'), (df_bad_win, 'Bad', 'tab:red')]:
        axes[1, 0].plot(df['window_start'], df['bead_width_cal_mm'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Bead Width (mm)')
    axes[1, 0].set_title('Calibrated Bead Width')
    axes[1, 0].legend()

    # Bead height
    for df, label, color in [(df_good_win, 'Good', 'tab:blue'), (df_bad_win, 'Bad', 'tab:red')]:
        axes[1, 1].plot(df['window_start'], df['bead_height_cal_mm'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Bead Height (mm)')
    axes[1, 1].set_title('Calibrated Bead Height')
    axes[1, 1].legend()

    # Thermal energy
    for df, label, color in [(df_good_win, 'Good', 'tab:blue'), (df_bad_win, 'Bad', 'tab:red')]:
        axes[1, 2].plot(df['window_start'], df['thermal_energy_haz_mean'], '.', alpha=0.5, color=color, label=label, markersize=3)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Thermal Energy (HAZ)')
    axes[1, 2].set_title('IR Thermal Energy Over Time')
    axes[1, 2].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved data_overview.png")


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]):
    """Plot correlation heatmap between features and targets."""
    cols = feature_cols + target_cols
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5,
                annot_kws={'size': 6})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")


def plot_regression_results(all_results: Dict, target_name: str):
    """Plot predicted vs actual for regression models."""
    n_groups = len(all_results)
    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 6 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)

    for i, (group_name, group_results) in enumerate(all_results.items()):
        for j, (model_name, res) in enumerate(group_results.items()):
            ax = axes[i, j]
            ax.scatter(res['y_true'], res['y_pred'], alpha=0.4, s=15, edgecolor='none')
            lims = [
                min(res['y_true'].min(), res['y_pred'].min()),
                max(res['y_true'].max(), res['y_pred'].max())
            ]
            ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name} - {group_name}\n'
                        f'R²={res["r2_mean"]:.4f}±{res["r2_std"]:.4f}  '
                        f'MAE={res["mae_mean"]:.4f}±{res["mae_std"]:.4f}')
            ax.legend()

    fig.suptitle(f'Regression: {target_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    safe_name = target_name.replace(' ', '_').replace('(', '').replace(')', '')
    fig.savefig(OUTPUT_DIR / f'regression_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved regression_{safe_name}.png")


def plot_roc_curves(all_clf_results: Dict):
    """Plot ROC curves for all feature groups and models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (group_name, group_results) in enumerate(all_clf_results.items()):
        ax = axes[i]
        for model_name, res in group_results.items():
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
            ax.plot(fpr, tpr, linewidth=2,
                   label=f'{model_name} (AUC={res["auc_mean"]:.3f}±{res["auc_std"]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {group_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved roc_curves.png")


def plot_feature_importance(all_clf_results: Dict):
    """Plot feature importance for all models and groups."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 24))

    for i, (group_name, group_results) in enumerate(all_clf_results.items()):
        for j, (model_name, res) in enumerate(group_results.items()):
            ax = axes[i, j]
            model = res['model']
            feat_names = res['feature_names']

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20

            # Shorten feature names
            short_names = [feat_names[idx].replace('_mean', '(μ)').replace('_std', '(σ)')
                          for idx in indices]

            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(short_names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name} - {group_name}', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved feature_importance.png")


def plot_shap_analysis(all_clf_results: Dict, X_dict: Dict[str, np.ndarray]):
    """Generate SHAP plots for model interpretability."""
    if not HAS_SHAP:
        logger.warning("SHAP not available, skipping SHAP plots.")
        return

    for group_name, group_results in all_clf_results.items():
        for model_name, res in group_results.items():
            model = res['model']
            feat_names = res['feature_names']
            X = X_dict[group_name]

            # Use a sample for SHAP (limit to 200 for speed)
            sample_size = min(200, X.shape[0])
            np.random.seed(RANDOM_SEED)
            sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_idx]

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                # For binary classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]  # Class 1 (bad bead)
                else:
                    shap_vals = shap_values

                # Summary plot
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_vals, X_sample,
                                 feature_names=[n.replace('_mean', '(μ)').replace('_std', '(σ)')
                                               for n in feat_names],
                                 show=False, max_display=20)
                plt.title(f'SHAP Summary - {model_name} ({group_name})', fontsize=13, fontweight='bold')
                plt.tight_layout()
                safe_group = group_name.replace(' ', '_')
                safe_model = model_name.replace(' ', '_')
                plt.savefig(OUTPUT_DIR / f'shap_{safe_group}_{safe_model}.png', dpi=150, bbox_inches='tight')
                plt.close('all')
                logger.info(f"Saved shap_{safe_group}_{safe_model}.png")

            except Exception as e:
                logger.error(f"SHAP failed for {model_name}/{group_name}: {e}")


def plot_confusion_matrices(all_clf_results: Dict):
    """Plot confusion matrices for classification results."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for i, (group_name, group_results) in enumerate(all_clf_results.items()):
        for j, (model_name, res) in enumerate(group_results.items()):
            ax = axes[i, j]
            cm = confusion_matrix(res['y_true'], res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name} - {group_name}\n'
                        f'Acc={res["acc_mean"]:.3f}±{res["acc_std"]:.3f}')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved confusion_matrices.png")


def plot_regression_feature_importance(all_reg_results: Dict, target_name: str):
    """Plot feature importance for regression models."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 24))

    for i, (group_name, group_results) in enumerate(all_reg_results.items()):
        for j, (model_name, res) in enumerate(group_results.items()):
            ax = axes[i, j]
            model = res['model']
            feat_names = res['feature_names']
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            short_names = [feat_names[idx].replace('_mean', '(μ)').replace('_std', '(σ)')
                          for idx in indices]

            ax.barh(range(len(indices)), importances[indices], color='darkorange')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(short_names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name} - {group_name}', fontsize=11, fontweight='bold')

    fig.suptitle(f'Regression Feature Importance: {target_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    safe_name = target_name.replace(' ', '_').replace('(', '').replace(')', '')
    fig.savefig(OUTPUT_DIR / f'reg_feat_importance_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved reg_feat_importance_{safe_name}.png")


def plot_shap_regression(all_reg_results: Dict, X_dict: Dict, target_name: str):
    """SHAP analysis for regression models."""
    if not HAS_SHAP:
        return

    for group_name, group_results in all_reg_results.items():
        # Use XGBoost model for SHAP
        if 'XGBoost' in group_results:
            res = group_results['XGBoost']
            model = res['model']
            feat_names = res['feature_names']
            X = X_dict[group_name]

            sample_size = min(200, X.shape[0])
            np.random.seed(RANDOM_SEED)
            sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_idx]

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample,
                                 feature_names=[n.replace('_mean', '(μ)').replace('_std', '(σ)')
                                               for n in feat_names],
                                 show=False, max_display=15)
                safe_target = target_name.replace(' ', '_').replace('(', '').replace(')', '')
                safe_group = group_name.replace(' ', '_')
                plt.title(f'SHAP - XGBoost Regression ({group_name}): {target_name}',
                         fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f'shap_reg_{safe_group}_{safe_target}.png',
                           dpi=150, bbox_inches='tight')
                plt.close('all')
                logger.info(f"Saved SHAP regression plot for {group_name}/{target_name}")
            except Exception as e:
                logger.error(f"SHAP regression failed: {e}")


# ============================================================
# Results Summary
# ============================================================

def generate_results_tables(all_clf_results: Dict, all_reg_results: Dict) -> str:
    """Generate results summary tables as markdown."""

    lines = []
    lines.append("# Classification Results: Bead Quality (Good vs Bad)\n")
    lines.append("| Feature Group | Model | Accuracy | F1 Score | AUC-ROC |")
    lines.append("|---|---|---|---|---|")

    for group_name, group_results in all_clf_results.items():
        for model_name, res in group_results.items():
            lines.append(
                f"| {group_name} | {model_name} | "
                f"{res['acc_mean']:.4f} ± {res['acc_std']:.4f} | "
                f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f} | "
                f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f} |"
            )

    lines.append("\n\n# Regression Results: Bead Geometry Prediction\n")

    for target_name, target_results in all_reg_results.items():
        lines.append(f"\n## {target_name}\n")
        lines.append("| Feature Group | Model | R² | MAE | RMSE |")
        lines.append("|---|---|---|---|---|")

        for group_name, group_results in target_results.items():
            for model_name, res in group_results.items():
                lines.append(
                    f"| {group_name} | {model_name} | "
                    f"{res['r2_mean']:.4f} ± {res['r2_std']:.4f} | "
                    f"{res['mae_mean']:.4f} ± {res['mae_std']:.4f} | "
                    f"{res['rmse_mean']:.4f} ± {res['rmse_std']:.4f} |"
                )

    return '\n'.join(lines)


# ============================================================
# Main Pipeline
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("WAAM Bead Geometry Prediction - ML Analysis")
    logger.info("=" * 60)

    # ---- Step 1: Load data ----
    logger.info("\n--- Step 1: Loading and filtering data ---")
    df_good = load_and_filter_data(str(GOOD_CSV), 'good')
    df_bad = load_and_filter_data(str(BAD_CSV), 'bad')

    # ---- Step 2: Create time windows ----
    logger.info("\n--- Step 2: Creating time windows ---")
    df_good_win = create_time_windows(df_good, WINDOW_SIZE_SEC, WINDOW_STEP_SEC)
    df_bad_win = create_time_windows(df_bad, WINDOW_SIZE_SEC, WINDOW_STEP_SEC)

    logger.info(f"  Good bead windows: {len(df_good_win)}")
    logger.info(f"  Bad bead windows: {len(df_bad_win)}")

    # Apply calibration
    df_good_win = apply_calibration(df_good_win)
    df_bad_win = apply_calibration(df_bad_win)

    # ---- Step 3: Handle data imbalance ----
    logger.info("\n--- Step 3: Handling data imbalance ---")
    # Good bead has fewer windows due to faster travel speed
    # Oversample good bead windows with slight noise injection
    n_good = len(df_good_win)
    n_bad = len(df_bad_win)
    logger.info(f"  Before balancing: Good={n_good}, Bad={n_bad}")

    if n_good < n_bad:
        # Oversample good with bootstrap + noise
        np.random.seed(RANDOM_SEED)
        n_needed = n_bad - n_good
        oversample_idx = np.random.choice(n_good, n_needed, replace=True)
        df_good_oversample = df_good_win.iloc[oversample_idx].copy()

        # Add small noise (1% of std) to numeric columns to avoid exact duplicates
        numeric_cols = df_good_oversample.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_std = df_good_win[col].std()
            if col_std > 0:
                noise = np.random.normal(0, 0.01 * col_std, len(df_good_oversample))
                df_good_oversample[col] = df_good_oversample[col].values + noise

        df_good_balanced = pd.concat([df_good_win, df_good_oversample], ignore_index=True)
        logger.info(f"  After balancing: Good={len(df_good_balanced)}, Bad={n_bad}")
    else:
        df_good_balanced = df_good_win

    # ---- Step 4: Combine dataset ----
    df_combined = pd.concat([df_good_balanced, df_bad_win], ignore_index=True)
    df_combined['bead_quality_label'] = (df_combined['bead_quality'] == 'bad').astype(int)
    logger.info(f"  Combined dataset: {len(df_combined)} samples")
    logger.info(f"  Class distribution: {df_combined['bead_quality_label'].value_counts().to_dict()}")

    # ---- Step 5: Data overview plots ----
    logger.info("\n--- Step 4: Generating data overview plots ---")
    plot_data_overview(df_good_win, df_bad_win)

    # Feature groups
    feature_groups = get_feature_groups(df_combined)
    for gname, feats in feature_groups.items():
        logger.info(f"  {gname}: {len(feats)} features")

    # Get all feature cols for correlation
    all_feat_cols = feature_groups['All Features Combined']
    target_cal_cols = ['bead_width_cal_mm', 'bead_height_cal_mm', 'bead_area_cal_mm2']
    plot_correlation_heatmap(df_combined, all_feat_cols, target_cal_cols)

    # ---- Step 6: Classification - Bead Quality ----
    logger.info("\n--- Step 5: Classification - Bead Quality Prediction ---")
    y_clf = df_combined['bead_quality_label'].values
    all_clf_results = {}
    X_clf_dict = {}

    for group_name, feat_cols in feature_groups.items():
        logger.info(f"\n  Feature Group: {group_name}")
        X = df_combined[feat_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_clf_dict[group_name] = X_scaled
        all_clf_results[group_name] = train_and_evaluate_classification(
            X_scaled, y_clf, feat_cols, group_name
        )

    # Classification plots
    plot_roc_curves(all_clf_results)
    plot_confusion_matrices(all_clf_results)
    plot_feature_importance(all_clf_results)
    plot_shap_analysis(all_clf_results, X_clf_dict)

    # ---- Step 7: Regression - Bead Geometry ----
    logger.info("\n--- Step 6: Regression - Bead Geometry Prediction ---")
    regression_targets = {
        'Bead Width (mm)': 'bead_width_cal_mm',
        'Bead Height (mm)': 'bead_height_cal_mm',
        'Bead Area (mm²)': 'bead_area_cal_mm2'
    }

    all_reg_results = {}
    for target_name, target_col in regression_targets.items():
        logger.info(f"\n  Target: {target_name}")
        all_reg_results[target_name] = {}
        X_reg_dict = {}

        y_reg = df_combined[target_col].values

        for group_name, feat_cols in feature_groups.items():
            logger.info(f"    Feature Group: {group_name}")
            X = df_combined[feat_cols].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_reg_dict[group_name] = X_scaled

            all_reg_results[target_name][group_name] = train_and_evaluate_regression(
                X_scaled, y_reg, feat_cols, target_name, group_name
            )

        # Regression plots
        plot_regression_results(all_reg_results[target_name], target_name)
        plot_regression_feature_importance(all_reg_results[target_name], target_name)
        plot_shap_regression(all_reg_results[target_name], X_reg_dict, target_name)

    # ---- Step 8: Save results ----
    logger.info("\n--- Step 7: Saving results ---")
    results_md = generate_results_tables(all_clf_results, all_reg_results)
    with open(OUTPUT_DIR / 'results_summary.md', 'w') as f:
        f.write(results_md)
    logger.info("Saved results_summary.md")

    # Save windowed data
    df_good_win.to_csv(OUTPUT_DIR / 'windowed_good.csv', index=False)
    df_bad_win.to_csv(OUTPUT_DIR / 'windowed_bad.csv', index=False)
    df_combined.to_csv(OUTPUT_DIR / 'windowed_combined.csv', index=False)
    logger.info("Saved windowed CSV files")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete! All outputs saved to: " + str(OUTPUT_DIR))
    logger.info("=" * 60)

    return all_clf_results, all_reg_results


if __name__ == '__main__':
    all_clf_results, all_reg_results = main()
