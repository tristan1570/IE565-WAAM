"""
WAAM Bead Geometry Prediction Pipeline v3 — ML Analysis & Visualization
========================================================================
Multi-modal process signal analysis for predicting weld bead geometry
(height, width, area) using Random Forest and XGBoost with 5-Fold CV.
All outputs saved to output_v2/ folder.
"""

import os, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"c:\Users\trist\OneDrive - University of Tennessee\Documents\newest_data")
OUTPUT_DIR = BASE_DIR / "output_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

BEAD_IDS = [1, 2, 3, 4, 5, 6]
WINDOW_SIZE_S = 0.5
OVERLAP_FRAC = 0.5
CURRENT_THRESHOLD = 50.0
RANDOM_SEED = 42
N_FOLDS = 5

TC_COLS = ['Channel_0(°C)', 'Channel_1(°C)', 'Channel_2(°C)', 'Channel_3(°C)']
ELECTRICAL_COLS = ['Current(A)', 'Voltage(V)']
FLIR_COLS = ['FLIR_melt_pool_area', 'FLIR_haz_width', 'FLIR_peak_temp_p95',
             'FLIR_thermal_gradient', 'FLIR_hot_pixel_count']
TARGET_COLS = ['target_height_mm', 'target_width_mm', 'target_area_mm2']
TARGET_SHORT = {'target_height_mm': 'height', 'target_width_mm': 'width', 'target_area_mm2': 'area'}
TARGET_UNITS = {'height': 'mm', 'width': 'mm', 'area': 'mm²'}

RF_PARAMS = {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5,
             'min_samples_leaf': 2, 'random_state': RANDOM_SEED, 'n_jobs': -1}
XGB_PARAMS = {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': RANDOM_SEED,
              'n_jobs': -1, 'verbosity': 0}

BEAD_COLORS = {1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a',
               4: '#984ea3', 5: '#ff7f00', 6: '#a65628'}
GROUP_COLORS_MAP = {'Current': '#1f77b4', 'Voltage': '#ff7f0e', 'Amplitude': '#2ca02c',
                    'Channel': '#d62728', 'FLIR': '#9467bd'}


# =============================================================================
# Data Loading & Windowing
# =============================================================================
def load_bead_data(bead_id):
    df = pd.read_csv(BASE_DIR / f"Bead{bead_id}" / "aligned_data.csv")
    df['bead_id'] = bead_id
    return df

def extract_window_features(window, bead_id):
    features = {}
    for col in ELECTRICAL_COLS:
        v = window[col].values
        features[f'{col}_mean'] = np.mean(v)
        features[f'{col}_std'] = np.std(v)
        features[f'{col}_min'] = np.min(v)
        features[f'{col}_max'] = np.max(v)
        features[f'{col}_range'] = np.max(v) - np.min(v)
    if 'Amplitude' in window.columns and bead_id != 1:
        v = window['Amplitude'].values
        for s in ['mean','std','min','max','range']:
            features[f'Amplitude_{s}'] = getattr(np, s if s != 'range' else 'ptp')(v) if s != 'range' else np.max(v)-np.min(v)
        features['Amplitude_mean'] = np.mean(v)
        features['Amplitude_std'] = np.std(v)
        features['Amplitude_min'] = np.min(v)
        features['Amplitude_max'] = np.max(v)
        features['Amplitude_range'] = np.max(v) - np.min(v)
    else:
        for s in ['mean','std','min','max','range']:
            features[f'Amplitude_{s}'] = np.nan
    for col in TC_COLS:
        v = window[col].values
        features[f'{col}_mean'] = np.mean(v)
        t = window['time'].values
        features[f'{col}_slope'] = (v[-1]-v[0])/(t[-1]-t[0]) if len(t)>1 and t[-1]!=t[0] else 0.0
    for col in FLIR_COLS:
        v = window[col].values
        features[f'{col}_mean'] = np.mean(v)
        features[f'{col}_std'] = np.std(v)
    return features

def create_windowed_dataset():
    all_windows = []
    for bead_id in BEAD_IDS:
        logger.info(f"Processing Bead {bead_id}...")
        df = load_bead_data(bead_id)
        df_active = df[df['Current(A)'].abs() > CURRENT_THRESHOLD].copy().reset_index(drop=True)
        if len(df_active) == 0: continue
        dt = df_active['time'].diff().median()
        spw = int(WINDOW_SIZE_S / dt)
        step = int(spw * (1 - OVERLAP_FRAC))
        cnt = 0
        for si in range(0, len(df_active) - spw + 1, step):
            w = df_active.iloc[si:si+spw]
            feat = extract_window_features(w, bead_id)
            for tc in TARGET_COLS:
                feat[tc] = w[tc].mean()
            feat['bead_id'] = bead_id
            feat['window_start_time'] = w['time'].iloc[0]
            feat['window_center_time'] = (w['time'].iloc[0] + w['time'].iloc[-1]) / 2
            all_windows.append(feat)
            cnt += 1
        logger.info(f"  Bead {bead_id}: {cnt} windows")
    return pd.DataFrame(all_windows)

def get_feature_groups(df):
    electrical = [c for c in df.columns if c.startswith(('Current(A)_', 'Voltage(V)_'))]
    acoustic = [c for c in df.columns if c.startswith('Amplitude_')]
    thermocouple = [c for c in df.columns if c.startswith('Channel_')]
    flir = [c for c in df.columns if c.startswith('FLIR_')]
    return {
        'Electrical': electrical,
        'Acoustic': acoustic,
        'Thermocouple': thermocouple,
        'FLIR': flir,
        'Elec+Acou': electrical + acoustic,
        'Elec+TC': electrical + thermocouple,
        'Elec+FLIR': electrical + flir,
        'All': electrical + acoustic + thermocouple + flir,
    }

def get_group_color(feat_name):
    for prefix, c in GROUP_COLORS_MAP.items():
        if feat_name.startswith(prefix):
            return c
    return '#888888'


# =============================================================================
# Model Training
# =============================================================================
def run_full_cv(df, feature_cols, target_col, model_type='xgb'):
    df_work = df.copy()
    actual = [c for c in feature_cols if c in df_work.columns]
    for c in actual:
        if c.startswith('Amplitude_'):
            df_work[c] = df_work[c].fillna(0)
    X = df_work[actual].values
    y = df_work[target_col].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    params = RF_PARAMS if model_type == 'rf' else XGB_PARAMS
    model_cls = RandomForestRegressor if model_type == 'rf' else xgb.XGBRegressor
    model = model_cls(**params)
    r2 = cross_val_score(model, X_s, y, cv=kf, scoring='r2')
    mae = -cross_val_score(model, X_s, y, cv=kf, scoring='neg_mean_absolute_error')
    rmse = np.sqrt(-cross_val_score(model, X_s, y, cv=kf, scoring='neg_mean_squared_error'))
    y_pred = cross_val_predict(model, X_s, y, cv=kf)
    model.fit(X_s, y)
    return {
        'r2_scores': r2, 'mae_scores': mae, 'rmse_scores': rmse,
        'r2_mean': r2.mean(), 'r2_std': r2.std(),
        'mae_mean': mae.mean(), 'mae_std': mae.std(),
        'rmse_mean': rmse.mean(), 'rmse_std': rmse.std(),
        'y_true': y, 'y_pred': y_pred,
        'bead_ids': df_work['bead_id'].values,
        'model': model, 'scaler': scaler,
        'feature_names': actual, 'X_scaled': X_s,
    }


# =============================================================================
# Plot 1: Data Overview — signals over time per bead
# =============================================================================
def plot_data_overview(df_windows):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('WAAM Process Data Overview (Time-Windowed, 6 Beads)', fontsize=16, fontweight='bold')

    plot_cols = [
        ('Current(A)_mean', 'Current (A)', axes[0,0]),
        ('Voltage(V)_mean', 'Voltage (V)', axes[0,1]),
        ('Amplitude_mean', 'Acoustic Amplitude', axes[0,2]),
        ('target_height_mm', 'Height (mm)', axes[1,0]),
        ('target_width_mm', 'Width (mm)', axes[1,1]),
        ('target_area_mm2', 'Area (mm²)', axes[1,2]),
    ]
    for col, ylabel, ax in plot_cols:
        for b in BEAD_IDS:
            sub = df_windows[df_windows['bead_id'] == b]
            vals = sub[col].values
            if col == 'Amplitude_mean' and b == 1:
                continue
            ax.plot(sub['window_center_time'], vals, '.', alpha=0.6,
                    color=BEAD_COLORS[b], label=f'Bead {b}', markersize=5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved data_overview.png")


# =============================================================================
# Plot 2: Full feature-feature correlation heatmap
# =============================================================================
def plot_full_correlation_heatmap(df_windows):
    fg = get_feature_groups(df_windows)
    all_feats = fg['All']
    actual = [c for c in all_feats if c in df_windows.columns] + list(TARGET_COLS)
    dfw = df_windows[actual].copy()
    for c in dfw.columns:
        dfw[c] = dfw[c].fillna(0)
    corr = dfw.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.3, annot_kws={'size': 6},
                vmin=-1, vmax=1)
    ax.set_title('Full Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap_full.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved correlation_heatmap_full.png")


# =============================================================================
# Plot 3: Feature-target correlation (top 20)
# =============================================================================
def plot_feature_target_correlation(df_windows):
    fg = get_feature_groups(df_windows)
    all_feats = [c for c in fg['All'] if c in df_windows.columns]
    dfw = df_windows.copy()
    for c in all_feats:
        dfw[c] = dfw[c].fillna(0)
    corr_data = []
    for f in all_feats:
        for t in TARGET_COLS:
            corr_data.append({'Feature': f, 'Target': TARGET_SHORT[t],
                              'Correlation': dfw[f].corr(dfw[t])})
    corr_df = pd.DataFrame(corr_data)
    pivot = corr_df.pivot_table(index='Feature', columns='Target', values='Correlation')
    pivot['abs_mean'] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values('abs_mean', ascending=True).drop('abs_mean', axis=1).tail(20)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                ax=ax, linewidths=0.5)
    ax.set_title('Top 20 Feature-Target Pearson Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_feature_target.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved correlation_feature_target.png")
    return corr_df


# =============================================================================
# Plot 4: Feature distribution boxplots by bead
# =============================================================================
def plot_feature_distributions(df_windows):
    top_feats = ['Current(A)_mean', 'Voltage(V)_std', 'Current(A)_std',
                 'Amplitude_mean', 'Channel_1(°C)_mean', 'FLIR_melt_pool_area_mean']
    actual = [f for f in top_feats if f in df_windows.columns]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Feature Distributions by Bead', fontsize=16, fontweight='bold')
    for idx, feat in enumerate(actual):
        ax = axes[idx // 3, idx % 3]
        data_by_bead = [df_windows[df_windows['bead_id']==b][feat].dropna().values for b in BEAD_IDS]
        bp = ax.boxplot(data_by_bead, labels=[f'B{b}' for b in BEAD_IDS], patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(BEAD_COLORS[BEAD_IDS[i]])
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
    for idx in range(len(actual), 6):
        axes[idx//3, idx%3].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved feature_distributions.png")


# =============================================================================
# Plot 5: Target distribution boxplots by bead
# =============================================================================
def plot_target_distributions(df_windows):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Bead Geometry Target Distributions by Bead', fontsize=15, fontweight='bold')
    for idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
        ax = axes[idx]
        data = [df_windows[df_windows['bead_id']==b][tc].values for b in BEAD_IDS]
        bp = ax.boxplot(data, labels=[f'B{b}' for b in BEAD_IDS], patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(BEAD_COLORS[BEAD_IDS[i]])
            patch.set_alpha(0.7)
        ax.set_title(f'{ts.title()} ({TARGET_UNITS[ts]})', fontsize=12, fontweight='bold')
        ax.set_ylabel(TARGET_UNITS[ts])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved target_distributions.png")


# =============================================================================
# Plot 6: Pairplot of targets
# =============================================================================
def plot_target_pairplot(df_windows):
    sub = df_windows[list(TARGET_COLS) + ['bead_id']].copy()
    sub.columns = ['Height (mm)', 'Width (mm)', 'Area (mm²)', 'bead_id']
    sub['Bead'] = sub['bead_id'].map(lambda x: f'Bead {int(x)}')
    bead_palette = {f'Bead {k}': v for k, v in BEAD_COLORS.items()}
    g = sns.pairplot(sub, hue='Bead', vars=['Height (mm)', 'Width (mm)', 'Area (mm²)'],
                     palette=bead_palette, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    g.figure.suptitle('Target Geometry Pairplot', fontsize=15, fontweight='bold', y=1.02)
    g.savefig(OUTPUT_DIR / 'target_pairplot.png', dpi=200, bbox_inches='tight')
    plt.close('all')
    logger.info("Saved target_pairplot.png")


# =============================================================================
# Plot 7: Per-group regression scatter (pred vs actual)
# =============================================================================
def plot_per_group_regression(all_results, target_short):
    groups = ['Electrical', 'Acoustic', 'Thermocouple', 'FLIR', 'All']
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 5*len(groups)))
    fig.suptitle(f'Regression: {target_short.title()} — Predicted vs Actual by Feature Group',
                 fontsize=16, fontweight='bold', y=1.0)
    for i, g in enumerate(groups):
        for j, mt in enumerate(['rf', 'xgb']):
            ax = axes[i, j]
            key = f"{g}_{target_short}_{mt}"
            if key not in all_results:
                ax.set_visible(False)
                continue
            res = all_results[key]
            yt, yp = res['y_true'], res['y_pred']
            bids = res['bead_ids']
            for b in sorted(set(bids)):
                m = bids == b
                ax.scatter(yt[m], yp[m], alpha=0.5, s=20, color=BEAD_COLORS[int(b)],
                           label=f'B{int(b)}', edgecolors='white', linewidth=0.3)
            mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{mt.upper()} — {g}\n'
                         f'R²={res["r2_mean"]:.3f}±{res["r2_std"]:.3f}  '
                         f'MAE={res["mae_mean"]:.3f}', fontsize=10)
            ax.legend(fontsize=6, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'regression_scatter_{target_short}.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved regression_scatter_{target_short}.png")


# =============================================================================
# Plot 8: R² comparison heatmap (all groups × all targets × both models)
# =============================================================================
def plot_r2_heatmap(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('R² Score Comparison Across Feature Groups', fontsize=16, fontweight='bold')
    group_order = ['Electrical','Acoustic','Thermocouple','FLIR','Elec+Acou','Elec+TC','Elec+FLIR','All']
    for ax_idx, model in enumerate(['RF', 'XGB']):
        ax = axes[ax_idx]
        sub = summary_df[summary_df['Model'] == model]
        pivot = sub.pivot_table(index='Feature Group', columns='Target', values='R2_mean')
        pivot = pivot.reindex([g for g in group_order if g in pivot.index])
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1.0,
                    ax=ax, linewidths=0.5, cbar_kws={'label': 'R²'})
        ax.set_title(f'{model}', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'r2_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved r2_heatmap.png")


# =============================================================================
# Plot 9: R² bar chart with error bars
# =============================================================================
def plot_r2_bar(summary_df):
    targets = ['height', 'width', 'area']
    group_order = ['Electrical','Acoustic','Thermocouple','FLIR','Elec+Acou','Elec+TC','Elec+FLIR','All']
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Model Performance by Feature Group (5-Fold CV)', fontsize=16, fontweight='bold')
    for ax_idx, target in enumerate(targets):
        ax = axes[ax_idx]
        sub = summary_df[summary_df['Target'] == target]
        x = np.arange(len(group_order))
        w = 0.35
        for mi, model in enumerate(['RF', 'XGB']):
            ms = sub[sub['Model'] == model]
            vals = [ms[ms['Feature Group']==g]['R2_mean'].values[0] if len(ms[ms['Feature Group']==g])>0 else 0 for g in group_order]
            errs = [ms[ms['Feature Group']==g]['R2_std'].values[0] if len(ms[ms['Feature Group']==g])>0 else 0 for g in group_order]
            ax.bar(x + mi*w - w/2, vals, w, yerr=errs, capsize=3,
                   label=model, color='#2196F3' if model=='RF' else '#FF9800', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title(f'{target.title()}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.set_ylim(bottom=-0.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'r2_bar_chart.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved r2_bar_chart.png")


# =============================================================================
# Plot 10: Predicted vs Actual — All features (clean version)
# =============================================================================
def plot_pred_vs_actual_all(all_results):
    for mt in ['rf', 'xgb']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f'Predicted vs Actual ({mt.upper()}, All Features, 5-Fold CV)',
                     fontsize=15, fontweight='bold')
        for ax_idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
            ax = axes[ax_idx]
            key = f"All_{ts}_{mt}"
            res = all_results[key]
            yt, yp, bids = res['y_true'], res['y_pred'], res['bead_ids']
            for b in sorted(set(bids)):
                m = bids == b
                ax.scatter(yt[m], yp[m], alpha=0.5, s=25, color=BEAD_COLORS[int(b)],
                           label=f'Bead {int(b)}', edgecolors='white', linewidth=0.3)
            mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            margin = (mx - mn) * 0.05
            ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'k--', lw=1.5)
            r2 = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.set_xlabel(f'Actual {ts.title()} ({TARGET_UNITS[ts]})')
            ax.set_ylabel(f'Predicted {ts.title()} ({TARGET_UNITS[ts]})')
            ax.set_title(f'{ts.title()}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=7, markerscale=1.5, loc='lower right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'pred_vs_actual_{mt}.png', dpi=200, bbox_inches='tight')
        plt.close()
    logger.info("Saved pred_vs_actual_*.png")


# =============================================================================
# Plot 11: Residual vs Predicted
# =============================================================================
def plot_residual_vs_predicted(all_results):
    for mt in ['rf', 'xgb']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Residuals vs Predicted ({mt.upper()}, All Features)', fontsize=15, fontweight='bold')
        for ax_idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
            ax = axes[ax_idx]
            res = all_results[f"All_{ts}_{mt}"]
            residuals = res['y_pred'] - res['y_true']
            bids = res['bead_ids']
            for b in sorted(set(bids)):
                m = bids == b
                ax.scatter(res['y_pred'][m], residuals[m], alpha=0.4, s=15,
                           color=BEAD_COLORS[int(b)], label=f'B{int(b)}')
            ax.axhline(0, color='red', linestyle='--', lw=1.5)
            ax.set_xlabel(f'Predicted {ts.title()}')
            ax.set_ylabel('Residual')
            ax.set_title(f'{ts.title()} (std={np.std(residuals):.3f})')
            ax.legend(fontsize=7, markerscale=2)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'residual_vs_predicted_{mt}.png', dpi=200, bbox_inches='tight')
        plt.close()
    logger.info("Saved residual_vs_predicted_*.png")


# =============================================================================
# Plot 12: Residual histograms
# =============================================================================
def plot_residual_histograms(all_results):
    for mt in ['rf', 'xgb']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Prediction Residual Distribution ({mt.upper()}, All Features)',
                     fontsize=15, fontweight='bold')
        for ax_idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
            ax = axes[ax_idx]
            res = all_results[f"All_{ts}_{mt}"]
            residuals = res['y_pred'] - res['y_true']
            ax.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(0, color='red', linestyle='--', lw=1.5)
            ax.set_xlabel(f'Residual ({TARGET_UNITS[ts]})')
            ax.set_ylabel('Count')
            ax.set_title(f'{ts.title()} (mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f})')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'residual_histogram_{mt}.png', dpi=200, bbox_inches='tight')
        plt.close()
    logger.info("Saved residual_histogram_*.png")


# =============================================================================
# Plot 13: Feature importance — All features, both models
# =============================================================================
def plot_feature_importance_all(all_results):
    for mt in ['rf', 'xgb']:
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        fig.suptitle(f'Top 15 Feature Importance ({mt.upper()}, All Features)',
                     fontsize=15, fontweight='bold')
        for ax_idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
            ax = axes[ax_idx]
            res = all_results[f"All_{ts}_{mt}"]
            model = res['model']
            feat_names = res['feature_names']
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            names = [feat_names[i] for i in idx]
            vals = [imp[i] for i in idx]
            colors = [get_group_color(n) for n in names]
            ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Importance')
            ax.set_title(f'{ts.title()}', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for l, c in GROUP_COLORS_MAP.items()]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10, bbox_to_anchor=(0.99, 0.95))
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.savefig(OUTPUT_DIR / f'feature_importance_{mt}.png', dpi=200, bbox_inches='tight')
        plt.close()
    logger.info("Saved feature_importance_*.png")


# =============================================================================
# Plot 14: Per-group feature importance for regression
# =============================================================================
def plot_per_group_feature_importance(all_results, target_short):
    groups = ['Electrical', 'Acoustic', 'Thermocouple', 'FLIR', 'All']
    fig, axes = plt.subplots(len(groups), 2, figsize=(18, 5*len(groups)))
    fig.suptitle(f'Feature Importance by Group: {target_short.title()}',
                 fontsize=16, fontweight='bold', y=1.0)
    for i, g in enumerate(groups):
        for j, mt in enumerate(['rf', 'xgb']):
            ax = axes[i, j]
            key = f"{g}_{target_short}_{mt}"
            if key not in all_results:
                ax.set_visible(False)
                continue
            res = all_results[key]
            imp = res['model'].feature_importances_
            fnames = res['feature_names']
            idx = np.argsort(imp)[::-1][:15]
            names = [fnames[ii].replace('_mean','(u)').replace('_std','(s)') for ii in idx]
            colors = [get_group_color(fnames[ii]) for ii in idx]
            ax.barh(range(len(idx)), imp[idx], color=colors, alpha=0.85)
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{mt.upper()} — {g}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'feat_importance_pergroup_{target_short}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feat_importance_pergroup_{target_short}.png")


# =============================================================================
# Plot 15: SHAP beeswarm — All features
# =============================================================================
def plot_shap_beeswarm(all_results):
    for ts in ['height', 'width', 'area']:
        res = all_results[f"All_{ts}_xgb"]
        model, X_s, fnames = res['model'], res['X_scaled'], res['feature_names']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_s, feature_names=fnames, show=False, max_display=20)
        plt.title(f'SHAP Feature Impact: {ts.title()} (XGBoost, All Features)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'shap_beeswarm_{ts}.png', dpi=200, bbox_inches='tight')
        plt.close('all')
    logger.info("Saved shap_beeswarm_*.png")


# =============================================================================
# Plot 16: SHAP bar — mean |SHAP|
# =============================================================================
def plot_shap_bar(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('Mean |SHAP| Feature Importance (XGBoost, All Features)',
                 fontsize=15, fontweight='bold')
    for ax_idx, ts in enumerate(['height', 'width', 'area']):
        ax = axes[ax_idx]
        res = all_results[f"All_{ts}_xgb"]
        model, X_s, fnames = res['model'], res['X_scaled'], res['feature_names']
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_s)
        mean_shap = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_shap)[::-1][:15]
        names = [fnames[i] for i in idx]
        vals = [mean_shap[i] for i in idx]
        colors = [get_group_color(n) for n in names]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'{ts.title()}', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_bar_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved shap_bar_importance.png")


# =============================================================================
# Plot 17: Per-group SHAP (XGBoost regression)
# =============================================================================
def plot_per_group_shap(all_results):
    for ts in ['height', 'width', 'area']:
        for g in ['Electrical', 'Thermocouple', 'FLIR', 'All']:
            key = f"{g}_{ts}_xgb"
            if key not in all_results:
                continue
            res = all_results[key]
            model, X_s, fnames = res['model'], res['X_scaled'], res['feature_names']
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_s)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(sv, X_s,
                                  feature_names=[n.replace('_mean','(u)').replace('_std','(s)') for n in fnames],
                                  show=False, max_display=15)
                safe_g = g.replace('+','_')
                plt.title(f'SHAP — XGBoost ({g}): {ts.title()}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f'shap_{safe_g}_{ts}.png', dpi=150, bbox_inches='tight')
                plt.close('all')
            except Exception as e:
                logger.error(f"SHAP failed {key}: {e}")
    logger.info("Saved per-group SHAP plots")


# =============================================================================
# Plot 18: Learning curves
# =============================================================================
def plot_learning_curves(df_windows):
    fg = get_feature_groups(df_windows)
    all_feats = [c for c in fg['All'] if c in df_windows.columns]
    dfw = df_windows.copy()
    for c in all_feats:
        if c.startswith('Amplitude_'):
            dfw[c] = dfw[c].fillna(0)
    X = dfw[all_feats].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Learning Curves (XGBoost, All Features)', fontsize=15, fontweight='bold')

    for ax_idx, (tc, ts) in enumerate(TARGET_SHORT.items()):
        ax = axes[ax_idx]
        y = dfw[tc].values
        model = xgb.XGBRegressor(**XGB_PARAMS)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_s, y, cv=5, scoring='r2',
            train_sizes=np.linspace(0.2, 1.0, 8), random_state=RANDOM_SEED, n_jobs=-1
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#2196F3', label='Train R²')
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='#2196F3')
        ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='#FF9800', label='CV R²')
        ax.fill_between(train_sizes,
                        test_scores.mean(axis=1) - test_scores.std(axis=1),
                        test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15, color='#FF9800')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{ts.title()}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'learning_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved learning_curves.png")


# =============================================================================
# Plot 19: CV fold boxplot
# =============================================================================
def plot_cv_boxplot(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Validation R² Distribution (All Features, 5-Fold)',
                 fontsize=15, fontweight='bold')
    for ax_idx, ts in enumerate(['height', 'width', 'area']):
        ax = axes[ax_idx]
        rf_r2 = all_results[f'All_{ts}_rf']['r2_scores']
        xgb_r2 = all_results[f'All_{ts}_xgb']['r2_scores']
        bp = ax.boxplot([rf_r2, xgb_r2], labels=['RF', 'XGB'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#FF9800'); bp['boxes'][1].set_alpha(0.7)
        ax.scatter([1, 2], [rf_r2.mean(), xgb_r2.mean()], s=80, zorder=5, marker='D',
                   color=['#2196F3','#FF9800'], edgecolors='black')
        ax.set_ylabel('R²')
        ax.set_title(f'{ts.title()}', fontsize=13, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cv_boxplot.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved cv_boxplot.png")


# =============================================================================
# Plot 20: Results dashboard
# =============================================================================
def plot_dashboard(summary_df):
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('WAAM Bead Geometry Prediction — Results Dashboard',
                 fontsize=18, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    group_order = ['Electrical','Acoustic','Thermocouple','FLIR','All']
    metrics_info = [('R2_mean', 'R2_std', 'R²'), ('MAE_mean', 'MAE_std', 'MAE'),
                    ('RMSE_mean', 'RMSE_std', 'RMSE')]
    for row_idx in range(2):
        for col_idx, ts in enumerate(['height', 'width', 'area']):
            if row_idx * 3 + col_idx >= 6:
                break
            metric_key, std_key, mlabel = metrics_info[min(row_idx, 1) + row_idx]
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sub = summary_df[summary_df['Target'] == ts]
            x = np.arange(len(group_order))
            w = 0.35
            for mi, model in enumerate(['RF', 'XGB']):
                ms = sub[sub['Model'] == model]
                vals = [ms[ms['Feature Group']==g][metric_key].values[0] if len(ms[ms['Feature Group']==g])>0 else 0 for g in group_order]
                errs = [ms[ms['Feature Group']==g][std_key].values[0] if len(ms[ms['Feature Group']==g])>0 else 0 for g in group_order]
                ax.bar(x + mi*w - w/2, vals, w, yerr=errs, capsize=2,
                       color='#2196F3' if model=='RF' else '#FF9800', alpha=0.85,
                       label=model if col_idx==0 and row_idx==0 else '')
            ax.set_xticks(x)
            ax.set_xticklabels(group_order, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(mlabel)
            ax.set_title(f'{ts.title()} — {mlabel}', fontsize=11, fontweight='bold')
    fig.legend(['RF', 'XGB'], loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.95))
    plt.savefig(OUTPUT_DIR / 'results_dashboard.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved results_dashboard.png")


# =============================================================================
# Summary table
# =============================================================================
def build_summary(all_results):
    rows = []
    for key, res in all_results.items():
        parts = key.rsplit('_', 2)
        mt, target, group = parts[-1].upper(), parts[-2], '_'.join(parts[:-2])
        rows.append({
            'Feature Group': group, 'Target': target, 'Model': mt,
            'R2_mean': res['r2_mean'], 'R2_std': res['r2_std'],
            'MAE_mean': res['mae_mean'], 'MAE_std': res['mae_std'],
            'RMSE_mean': res['rmse_mean'], 'RMSE_std': res['rmse_std'],
        })
    return pd.DataFrame(rows).sort_values(['Target', 'Feature Group', 'Model'])


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("WAAM ML Pipeline v3 — ML Analysis & Visualization")
    logger.info("=" * 70)

    # Phase 1-2: Windowing
    logger.info("\n[Phase 1-2] Windowing...")
    df_windows = create_windowed_dataset()
    df_windows.to_csv(OUTPUT_DIR / 'windowed_features.csv', index=False)

    # Phase 3-4: All experiments
    logger.info("\n[Phase 3-4] Experiments...")
    fg = get_feature_groups(df_windows)
    all_results = {}
    for gname, fcols in fg.items():
        for tc, ts in TARGET_SHORT.items():
            for mt in ['rf', 'xgb']:
                key = f"{gname}_{ts}_{mt}"
                logger.info(f"  {key}...")
                all_results[key] = run_full_cv(df_windows, fcols, tc, mt)

    summary_df = build_summary(all_results)
    summary_df.to_csv(OUTPUT_DIR / 'cv_summary.csv', index=False)

    # Phase 5-6: All plots
    logger.info("\n[Phase 5-6] Generating all plots...")
    plot_data_overview(df_windows)
    plot_full_correlation_heatmap(df_windows)
    corr_df = plot_feature_target_correlation(df_windows)
    plot_feature_distributions(df_windows)
    plot_target_distributions(df_windows)
    plot_target_pairplot(df_windows)
    plot_r2_heatmap(summary_df)
    plot_r2_bar(summary_df)
    plot_pred_vs_actual_all(all_results)
    for ts in ['height', 'width', 'area']:
        plot_per_group_regression(all_results, ts)
        plot_per_group_feature_importance(all_results, ts)
    plot_residual_vs_predicted(all_results)
    plot_residual_histograms(all_results)
    plot_feature_importance_all(all_results)
    plot_shap_beeswarm(all_results)
    plot_shap_bar(all_results)
    plot_per_group_shap(all_results)
    plot_learning_curves(df_windows)
    plot_cv_boxplot(all_results)
    plot_dashboard(summary_df)

    logger.info("\n" + "=" * 70)
    logger.info("Complete! Outputs: " + str(OUTPUT_DIR))
    logger.info("=" * 70)

    # Print summary
    print("\n" + "=" * 70)
    print("BEST RESULTS (XGBoost, All Features)")
    print("=" * 70)
    for ts in ['height','width','area']:
        res = all_results[f'All_{ts}_xgb']
        print(f"  {ts.upper()}: R²={res['r2_mean']:.3f}±{res['r2_std']:.3f}  "
              f"MAE={res['mae_mean']:.3f}  RMSE={res['rmse_mean']:.3f}")


if __name__ == '__main__':
    main()
