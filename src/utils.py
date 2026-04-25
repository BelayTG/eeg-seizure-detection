"""
utils.py
--------
Shared utility functions for EEG analysis notebooks.

All notebooks import from this module to avoid code duplication.

Usage
-----
    import sys
    sys.path.insert(0, os.path.join("..", "src"))
    from utils import build_file_index, load_epoch, get_per_animal_band_power, PATHS, COLORS
"""

import os
import gc
import numpy as np
import pandas as pd
import pyabf
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Study paths — single source of truth for all notebooks
# ---------------------------------------------------------------------------

BASE = r"C:\Users\belay\OneDrive\Desktop\EEG analysis"

PATHS = {
    "3m": {
        "WT": os.path.join(BASE, "Baseline EEG", "WT"),
        "KO": os.path.join(BASE, "Baseline EEG", "KO"),
        "manifest_wt": "abf_files_start_times_wt.xlsx",
        "manifest_ko": "abf_files_start_times_ko.xlsx",
    },
    "4m_KA": {
        "WT": os.path.join(BASE, "KA EEG", "WT"),
        "KO": os.path.join(BASE, "KA EEG", "KO"),
    },
    "6m": {
        "WT": os.path.join(BASE, "Six Months EEG", "WT"),
        "KO": os.path.join(BASE, "Six Months EEG", "KO"),
        "manifest_wt": "abf_files_start_times_wt_6m.xlsx",
        "manifest_ko": "abf_files_start_times_ko_6m.xlsx",
    },
    "12m": {
        "WT": os.path.join(BASE, "One Year EEG", "WT"),
        "KO": os.path.join(BASE, "One Year EEG", "KO"),
        "manifest_wt": "abf_files_start_times_wt_1y.xlsx",
        "manifest_ko": "abf_files_start_times_ko_1y.xlsx",
    },
}

# Animals excluded from analysis with documented reasons
EXCLUDED_ANIMALS = {
    "21507": "Body weight loss >20% during recording period — illness unrelated to genotype"
}

# Consistent colors across all notebooks
COLORS = {"WT": "#378ADD", "KO": "#D85A30"}

# Standard matplotlib settings
PLT_STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def build_file_index(data_dir: str) -> dict:
    """
    Recursively find all ABF files in a directory.

    Returns
    -------
    dict : filename -> full path
    """
    index = {}
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".abf"):
                index[fname] = os.path.join(root, fname)
    return index


def load_epoch(
    file_path: str,
    start_min: float,
    duration_min: float = 0.5,
    channel: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Load a single epoch from an ABF file.

    Loads only the required samples rather than the full recording,
    keeping memory usage low for long (24h) recordings.

    Parameters
    ----------
    file_path : str
        Path to the .abf file.
    start_min : float
        Epoch start time in minutes.
    duration_min : float
        Epoch duration in minutes (default 0.5 = 30 seconds).
    channel : int
        Channel index (0 = CA3, 1 = cortex).

    Returns
    -------
    epoch : np.ndarray
        Voltage array in mV.
    fs : float
        Sampling rate in Hz.
    """
    abf = pyabf.ABF(file_path)
    fs = abf.dataRate
    start_idx = int(start_min * 60 * fs)
    end_idx = int((start_min + duration_min) * 60 * fs)
    abf.setSweep(0, channel=channel)
    epoch = abf.sweepY[start_idx:end_idx].copy()
    del abf
    gc.collect()
    return epoch, fs


def load_manifest(data_dir: str, manifest_filename: str) -> pd.DataFrame:
    """Load and validate an epoch manifest Excel file."""
    path = os.path.join(data_dir, manifest_filename)
    df = pd.read_excel(path)
    assert "File" in df.columns, "Manifest missing 'File' column"
    assert "Start_Times" in df.columns, "Manifest missing 'Start_Times' column"
    return df


def parse_start_times(start_times_str: str) -> list[float]:
    """Parse comma-separated start times string into list of floats."""
    return [
        float(t.strip())
        for t in str(start_times_str).split(",")
        if t.strip()
    ]


# ---------------------------------------------------------------------------
# Per-animal band power
# ---------------------------------------------------------------------------

def get_per_animal_band_power(
    data_dir: str,
    manifest_filename: str,
    channel: int = 0,
    epoch_duration_min: float = 0.5,
    excluded_animals: dict | None = None,
) -> pd.DataFrame:
    """
    Compute mean band power per animal from epoch manifest.

    Uses 1 Hz frequency resolution (nperseg = sampling rate).
    Statistical unit = one animal (epochs averaged per animal).
    Animal ID = first 5 characters of filename.

    Parameters
    ----------
    data_dir : str
        Directory containing ABF files and manifest.
    manifest_filename : str
        Name of the Excel manifest file.
    channel : int
        Channel index (0 = CA3, 1 = cortex).
    epoch_duration_min : float
        Duration of each epoch in minutes.
    excluded_animals : dict, optional
        Dictionary of animal_id -> reason for exclusion.
        Defaults to the study-wide EXCLUDED_ANIMALS.

    Returns
    -------
    pd.DataFrame
        One row per animal with columns:
        animal_id, delta, theta, alpha, beta, gamma
    """
    from detection import compute_psd, compute_band_power

    if excluded_animals is None:
        excluded_animals = EXCLUDED_ANIMALS

    manifest = load_manifest(data_dir, manifest_filename)
    file_index = build_file_index(data_dir)
    animal_powers = {}

    for _, row in manifest.iterrows():
        fname = row["File"]
        animal_id = fname[:5]

        if animal_id in excluded_animals:
            continue
        if fname not in file_index:
            continue

        start_times = parse_start_times(row["Start_Times"])

        if animal_id not in animal_powers:
            animal_powers[animal_id] = []

        for start_min in start_times:
            try:
                epoch, fs = load_epoch(
                    file_index[fname], start_min,
                    duration_min=epoch_duration_min,
                    channel=channel,
                )
                freq, psd = compute_psd(epoch, fs, nperseg=int(fs))
                bp = compute_band_power(freq, psd)
                animal_powers[animal_id].append(bp)
                del epoch
                gc.collect()
            except Exception:
                pass

    rows = []
    for animal_id, bp_list in animal_powers.items():
        if bp_list:
            mean_bp = {k: np.mean([b[k] for b in bp_list]) for k in bp_list[0]}
            mean_bp["animal_id"] = animal_id
            rows.append(mean_bp)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compare_groups(
    wt_values: pd.Series,
    ko_values: pd.Series,
    label: str = "",
) -> dict:
    """
    Run Mann-Whitney U test and return formatted results.

    Parameters
    ----------
    wt_values : pd.Series
        Values for WT group.
    ko_values : pd.Series
        Values for KO group.
    label : str
        Label for printing.

    Returns
    -------
    dict with keys: stat, pval, sig, wt_mean, wt_sem, ko_mean, ko_sem
    """
    stat, pval = mannwhitneyu(wt_values, ko_values, alternative="two-sided")
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

    result = {
        "stat": stat,
        "pval": pval,
        "sig": sig,
        "wt_mean": wt_values.mean(),
        "wt_sem": wt_values.sem(),
        "ko_mean": ko_values.mean(),
        "ko_sem": ko_values.sem(),
        "wt_n": len(wt_values),
        "ko_n": len(ko_values),
    }

    if label:
        print(f"{label}:")
        print(f"  WT  mean ± SEM: {result['wt_mean']:.4f} ± {result['wt_sem']:.4f} (n={result['wt_n']})")
        print(f"  KO  mean ± SEM: {result['ko_mean']:.4f} ± {result['ko_sem']:.4f} (n={result['ko_n']})")
        print(f"  Mann-Whitney U: stat={stat:.1f}, p={pval:.4f} ({sig})")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_group_comparison(
    ax,
    wt_values: pd.Series,
    ko_values: pd.Series,
    stat_result: dict,
    ylabel: str = "",
    title: str = "",
) -> None:
    """
    Plot individual data points + mean ± SEM for WT vs KO comparison.

    Parameters
    ----------
    ax : matplotlib Axes
    wt_values : pd.Series
    ko_values : pd.Series
    stat_result : dict — output of compare_groups()
    ylabel : str
    title : str
    """
    for label, values in [("WT", wt_values), ("KO", ko_values)]:
        ax.scatter(
            [label] * len(values), values,
            color=COLORS[label], alpha=0.7, s=50, zorder=3
        )
        ax.errorbar(
            label, values.mean(), yerr=values.sem(),
            fmt="_", markersize=28, markeredgewidth=2.5,
            color=COLORS[label], capsize=5, capthick=2
        )

    sig = stat_result["sig"]
    pval = stat_result["pval"]
    ax.annotate(
        f"{sig}\np={pval:.3f}",
        xy=(0.5, 0.93), xycoords="axes fraction",
        ha="center", fontsize=10
    )

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def save_figure(fig, filename: str, figures_dir: str) -> None:
    """Save figure to figures directory."""
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def check_paths() -> None:
    """Verify all data paths exist and print status."""
    print("Checking data paths:")
    all_ok = True
    for timepoint, dirs in PATHS.items():
        wt_ok = os.path.exists(dirs["WT"])
        ko_ok = os.path.exists(dirs["KO"])
        status = "OK" if (wt_ok and ko_ok) else "MISSING"
        if not (wt_ok and ko_ok):
            all_ok = False
        print(f"  {timepoint}: WT={'OK' if wt_ok else 'NOT FOUND'} | KO={'OK' if ko_ok else 'NOT FOUND'}")
    if all_ok:
        print("All paths OK.")
    else:
        print("WARNING: Some paths not found. Check BASE directory.")
