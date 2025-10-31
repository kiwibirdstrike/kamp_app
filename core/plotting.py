# core/plotting.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_matplotlib_grid(df: pd.DataFrame, time_col: str, y_cols: list[str], n_cols: int = 4, hour_interval: int = 6):
    n = len(y_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), squeeze=False)

    for i, col in enumerate(y_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        y = pd.to_numeric(df[col], errors="coerce")
        m = df[time_col].notna() & y.notna()
        ax.plot(df.loc[m, time_col], y.loc[m], linewidth=1)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
        except Exception:
            pass

    # 남는 축 off
    filled = n
    total = n_rows * n_cols
    for j in range(filled, total):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_numeric_feature_with_thresholds(df: pd.DataFrame, feature: str, thresholds: list[float], bins: int = 40):
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = pd.to_numeric(df[feature], errors="coerce")
    vals = vals[vals.notna()]
    ax.hist(vals, bins=bins, alpha=0.6)
    for thr in thresholds:
        ax.axvline(thr, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{feature} — splits")
    ax.set_xlabel(feature)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
