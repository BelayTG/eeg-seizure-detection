"""
seizure_detection.py
--------------------
Automated detection of seizures and interictal epileptiform discharges (IEDs)
from kainic acid-induced EEG recordings.

Recording structure:
    - Two consecutive 2h ABF files per mouse
    - All WT files in one directory, all KO files in another
    - Files sorted alphabetically and paired sequentially

Detection pipeline:
    1. Load and concatenate both recordings per mouse
    2. Artifact rejection (voltage range filtering)
    3. Adaptive baseline estimation
    4. IED detection (brief high-amplitude transients)
    5. Seizure detection (sustained ictal activity >5 seconds)
    6. Event classification (IED vs seizure)
    7. Per-mouse metric computation

Metrics computed per mouse:
    - IED count and rate (events/min)
    - Seizure count
    - Seizure burden (% time in seizure)
    - Mean and total seizure duration
    - Mean inter-IED interval
    - Latency to first seizure

Usage
-----
    from seizure_detection import pair_files, process_mouse, process_group
    
    wt_pairs = pair_files(wt_dir)
    results_wt = process_group(wt_dir, wt_pairs, group="WT")
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------

def pair_files(data_dir: str) -> list[tuple[str, str]]:
    """
    Pair consecutive ABF files from a directory.
    
    Files are sorted alphabetically and paired sequentially:
    file[0] + file[1] = mouse 1, file[2] + file[3] = mouse 2, etc.

    Parameters
    ----------
    data_dir : str — directory containing ABF files

    Returns
    -------
    list of (file1, file2) tuples — one tuple per mouse
    """
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".abf")])
    pairs = []
    for i in range(0, len(files) - 1, 2):
        pairs.append((files[i], files[i + 1]))
    if len(files) % 2 != 0:
        print(f"  Warning: {files[-1]} is unpaired — excluded")
    return pairs


def get_mouse_id(file1: str, file2: str) -> str:
    """
    Generate a mouse ID from paired filenames.
    Uses the first 7 characters of file1 (date prefix).
    """
    return file1[:7]


# ---------------------------------------------------------------------------
# Signal loading and preprocessing
# ---------------------------------------------------------------------------

def load_abf_channel(file_path: str, channel: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Load voltage and time from an ABF file."""
    abf = pyabf.ABF(file_path)
    abf.setSweep(0, channel=channel)
    time = abf.sweepX.copy()
    voltage = abf.sweepY.copy()
    fs = abf.dataRate
    del abf
    gc.collect()
    return time, voltage, fs


def remove_artifacts(
    time: np.ndarray,
    voltage: np.ndarray,
    v_min: float = -10.0,
    v_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples outside physiological voltage range."""
    valid = (voltage >= v_min) & (voltage <= v_max)
    return time[valid], voltage[valid]


def estimate_baseline(
    voltage: np.ndarray,
    fs: float,
    duration_s: float = 3600,
    coverage: float = 0.97,
) -> float:
    """
    Estimate adaptive baseline threshold from initial recording segment.
    Returns threshold T such that coverage% of baseline falls below T.
    """
    n = int(min(duration_s * fs, len(voltage)))
    baseline_window = voltage[:n]
    threshold = 0.01
    increment = 0.0005
    while True:
        if np.sum(baseline_window < threshold) / len(baseline_window) >= coverage:
            return float(threshold)
        threshold += increment


def concatenate_recordings(
    file1_path: str,
    file2_path: str,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load and concatenate two 2h recordings into a single 4h trace.

    Parameters
    ----------
    file1_path, file2_path : str — paths to the two ABF files
    channel : int — channel to load

    Returns
    -------
    time : np.ndarray — concatenated time array (seconds)
    voltage : np.ndarray — concatenated voltage array (mV)
    fs : float — sampling rate in Hz
    """
    time1, voltage1, fs1 = load_abf_channel(file1_path, channel)
    time2, voltage2, fs2 = load_abf_channel(file2_path, channel)

    # Offset second recording time
    time_offset = time1[-1] + (1.0 / fs1)
    time2_offset = time2 + time_offset

    time = np.concatenate([time1, time2_offset])
    voltage = np.concatenate([voltage1, voltage2])

    del time1, voltage1, time2, voltage2
    gc.collect()

    return time, voltage, fs1


# ---------------------------------------------------------------------------
# IED detection
# ---------------------------------------------------------------------------

def detect_ieds(
    time: np.ndarray,
    voltage: np.ndarray,
    fs: float,
    lower_threshold: float,
    upper_threshold: float = 1.5,
    min_prominence: float = 0.2,
    max_width_s: float = 0.2,
    min_interval_s: float = 0.1,
) -> pd.DataFrame:
    """
    Detect interictal epileptiform discharges (IEDs).

    IEDs are brief (<200ms), high-amplitude spike-wave complexes
    occurring between seizures. Detection uses amplitude threshold,
    prominence, width, and refractory period criteria.

    Parameters
    ----------
    time : np.ndarray — time in seconds
    voltage : np.ndarray — voltage in mV
    fs : float — sampling rate in Hz
    lower_threshold : float — minimum peak amplitude (typically 2x baseline)
    upper_threshold : float — maximum peak amplitude (default 1.5 mV)
    min_prominence : float — minimum peak prominence (default 0.2 mV)
    max_width_s : float — maximum event width in seconds (default 0.2s)
    min_interval_s : float — refractory period in seconds (default 0.1s)

    Returns
    -------
    pd.DataFrame with columns: time_s, voltage_mV, prominence
    """
    peaks, properties = find_peaks(
        voltage,
        height=(lower_threshold, upper_threshold),
        prominence=min_prominence,
        width=(None, max_width_s * fs),
        distance=int(min_interval_s * fs),
    )

    if len(peaks) == 0:
        return pd.DataFrame(columns=["time_s", "voltage_mV", "prominence"])

    return pd.DataFrame({
        "time_s": time[peaks],
        "voltage_mV": voltage[peaks],
        "prominence": properties["prominences"],
    })


# ---------------------------------------------------------------------------
# Seizure detection
# ---------------------------------------------------------------------------

def detect_seizures(
    time: np.ndarray,
    voltage: np.ndarray,
    fs: float,
    ieds: pd.DataFrame,
    min_seizure_duration_s: float = 5.0,
    max_ied_gap_s: float = 2.0,
    min_ieds_per_seizure: int = 5,
) -> pd.DataFrame:
    """
    Detect sustained seizures by clustering dense IED activity.

    A seizure is defined as a cluster of IEDs where:
    - At least min_ieds_per_seizure events occur
    - Gap between consecutive events < max_ied_gap_s
    - Total cluster duration > min_seizure_duration_s

    This approach identifies both convulsive and non-convulsive seizures
    without requiring manual threshold adjustment.

    Parameters
    ----------
    time : np.ndarray
    voltage : np.ndarray
    fs : float
    ieds : pd.DataFrame — output of detect_ieds()
    min_seizure_duration_s : float — minimum seizure duration (default 5s)
    max_ied_gap_s : float — maximum gap between IEDs within a seizure (default 2s)
    min_ieds_per_seizure : int — minimum IEDs to constitute a seizure (default 5)

    Returns
    -------
    pd.DataFrame with columns:
        onset_s, offset_s, duration_s, n_ieds, mean_amplitude_mV
    """
    if len(ieds) < min_ieds_per_seizure:
        return pd.DataFrame(columns=[
            "onset_s", "offset_s", "duration_s", "n_ieds", "mean_amplitude_mV"
        ])

    ied_times = ieds["time_s"].values
    seizures = []
    cluster_start = ied_times[0]
    cluster_events = [ied_times[0]]

    for i in range(1, len(ied_times)):
        gap = ied_times[i] - ied_times[i - 1]
        if gap <= max_ied_gap_s:
            cluster_events.append(ied_times[i])
        else:
            # End of cluster — check if it qualifies as seizure
            duration = cluster_events[-1] - cluster_events[0]
            if (duration >= min_seizure_duration_s and
                    len(cluster_events) >= min_ieds_per_seizure):
                cluster_ieds = ieds[
                    (ieds["time_s"] >= cluster_events[0]) &
                    (ieds["time_s"] <= cluster_events[-1])
                ]
                seizures.append({
                    "onset_s": cluster_events[0],
                    "offset_s": cluster_events[-1],
                    "duration_s": duration,
                    "n_ieds": len(cluster_events),
                    "mean_amplitude_mV": cluster_ieds["voltage_mV"].mean(),
                })
            # Start new cluster
            cluster_events = [ied_times[i]]

    # Check last cluster
    if len(cluster_events) >= min_ieds_per_seizure:
        duration = cluster_events[-1] - cluster_events[0]
        if duration >= min_seizure_duration_s:
            cluster_ieds = ieds[
                (ieds["time_s"] >= cluster_events[0]) &
                (ieds["time_s"] <= cluster_events[-1])
            ]
            seizures.append({
                "onset_s": cluster_events[0],
                "offset_s": cluster_events[-1],
                "duration_s": duration,
                "n_ieds": len(cluster_events),
                "mean_amplitude_mV": cluster_ieds["voltage_mV"].mean(),
            })

    return pd.DataFrame(seizures) if seizures else pd.DataFrame(
        columns=["onset_s", "offset_s", "duration_s", "n_ieds", "mean_amplitude_mV"]
    )


# ---------------------------------------------------------------------------
# Per-mouse metrics
# ---------------------------------------------------------------------------

def compute_mouse_metrics(
    ieds: pd.DataFrame,
    seizures: pd.DataFrame,
    recording_duration_s: float,
) -> dict:
    """
    Compute summary metrics for one mouse from detected events.

    Parameters
    ----------
    ieds : pd.DataFrame — detected IEDs
    seizures : pd.DataFrame — detected seizures
    recording_duration_s : float — total recording duration

    Returns
    -------
    dict of metrics
    """
    duration_min = recording_duration_s / 60.0

    # IED metrics
    n_ieds = len(ieds)
    ied_rate = n_ieds / duration_min if duration_min > 0 else 0.0

    if n_ieds > 1:
        ied_intervals = np.diff(ieds["time_s"].values)
        mean_ied_interval = float(ied_intervals.mean())
    else:
        mean_ied_interval = 0.0

    # Seizure metrics
    n_seizures = len(seizures)

    if n_seizures > 0:
        total_seizure_dur = float(seizures["duration_s"].sum())
        mean_seizure_dur = float(seizures["duration_s"].mean())
        seizure_burden = 100.0 * total_seizure_dur / recording_duration_s
        first_seizure_min = float(seizures["onset_s"].min()) / 60.0
    else:
        total_seizure_dur = 0.0
        mean_seizure_dur = 0.0
        seizure_burden = 0.0
        first_seizure_min = np.nan

    return {
        "n_ieds": n_ieds,
        "ied_rate_per_min": round(ied_rate, 4),
        "mean_ied_interval_s": round(mean_ied_interval, 4),
        "n_seizures": n_seizures,
        "total_seizure_dur_s": round(total_seizure_dur, 2),
        "mean_seizure_dur_s": round(mean_seizure_dur, 2),
        "seizure_burden_pct": round(seizure_burden, 4),
        "first_seizure_min": round(first_seizure_min, 2) if not np.isnan(first_seizure_min) else np.nan,
        "recording_duration_min": round(duration_min, 2),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_mouse(
    file1_path: str,
    file2_path: str,
    mouse_id: str,
    channel: int = 0,
) -> dict:
    """
    Run full detection pipeline on one mouse (two 2h recordings).

    Parameters
    ----------
    file1_path, file2_path : str — paths to the two ABF files
    mouse_id : str — mouse identifier
    channel : int — EEG channel (0 = CA3, 1 = cortex)

    Returns
    -------
    dict containing:
        - mouse_id
        - all metrics from compute_mouse_metrics()
        - ieds DataFrame
        - seizures DataFrame
    """
    # Load and concatenate
    time, voltage, fs = concatenate_recordings(file1_path, file2_path, channel)

    # Artifact rejection
    time, voltage = remove_artifacts(time, voltage)

    # Baseline estimation
    baseline = estimate_baseline(voltage, fs)
    lower_threshold = 2.0 * baseline

    # IED detection
    ieds = detect_ieds(time, voltage, fs, lower_threshold=lower_threshold)

    # Seizure detection
    seizures = detect_seizures(time, voltage, fs, ieds)

    # Metrics
    metrics = compute_mouse_metrics(ieds, seizures, recording_duration_s=len(time) / fs)
    metrics["mouse_id"] = mouse_id
    metrics["baseline_mV"] = round(baseline, 6)
    metrics["threshold_mV"] = round(lower_threshold, 6)

    del time, voltage
    gc.collect()

    return {**metrics, "ieds": ieds, "seizures": seizures}


def process_group(
    data_dir: str,
    group: str = "WT",
    channel: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Process all mice in a group directory.

    Parameters
    ----------
    data_dir : str — directory with ABF files
    group : str — group label ('WT' or 'KO')
    channel : int — EEG channel
    verbose : bool — print progress

    Returns
    -------
    summary : pd.DataFrame — one row per mouse with all metrics
    details : dict — mouse_id -> {ieds, seizures} DataFrames
    """
    pairs = pair_files(data_dir)
    if verbose:
        print(f"  {len(pairs)} mice found in {group}")

    rows = []
    details = {}

    for i, (file1, file2) in enumerate(pairs):
        mouse_id = get_mouse_id(file1, file2)
        file1_path = os.path.join(data_dir, file1)
        file2_path = os.path.join(data_dir, file2)

        if verbose:
            print(f"  Processing mouse {i+1}/{len(pairs)}: {mouse_id} ({file1} + {file2})")

        try:
            result = process_mouse(file1_path, file2_path, mouse_id, channel)
            details[mouse_id] = {
                "ieds": result.pop("ieds"),
                "seizures": result.pop("seizures"),
            }
            result["group"] = group
            rows.append(result)

            if verbose:
                print(f"    IEDs: {result['n_ieds']} | Rate: {result['ied_rate_per_min']:.2f}/min | "
                      f"Seizures: {result['n_seizures']} | Burden: {result['seizure_burden_pct']:.2f}%")

        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")

    summary = pd.DataFrame(rows)
    return summary, details


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eeg_with_events(
    file1_path: str,
    file2_path: str,
    ieds: pd.DataFrame,
    seizures: pd.DataFrame,
    mouse_id: str,
    channel: int = 0,
    plot_duration_min: float = 30.0,
    save_path: str | None = None,
) -> None:
    """
    Plot EEG trace with detected IEDs and seizures overlaid.

    Parameters
    ----------
    file1_path, file2_path : str
    ieds : pd.DataFrame
    seizures : pd.DataFrame
    mouse_id : str
    channel : int
    plot_duration_min : float — how many minutes to plot (default 30)
    save_path : str, optional
    """
    time, voltage, fs = concatenate_recordings(file1_path, file2_path, channel)
    time, voltage = remove_artifacts(time, voltage)

    end_idx = int(plot_duration_min * 60 * fs)
    t_plot = time[:end_idx]
    v_plot = voltage[:end_idx]

    ieds_plot = ieds[ieds["time_s"] <= t_plot[-1]]
    seizures_plot = seizures[seizures["onset_s"] <= t_plot[-1]]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_plot / 60, v_plot, color="#2C2C2A", lw=0.3, label="EEG")

    if len(ieds_plot) > 0:
        ax.scatter(ieds_plot["time_s"] / 60, ieds_plot["voltage_mV"],
                   color="#D85A30", s=12, zorder=5,
                   label=f"IEDs (n={len(ieds_plot)} shown)")

    for _, szr in seizures_plot.iterrows():
        ax.axvspan(szr["onset_s"] / 60, szr["offset_s"] / 60,
                   color="#59A14F", alpha=0.25, label="_")

    if len(seizures_plot) > 0:
        ax.axvspan(0, 0, color="#59A14F", alpha=0.4, label=f"Seizures (n={len(seizures_plot)} shown)")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"Mouse {mouse_id} — KA EEG (first {plot_duration_min:.0f} min)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    del time, voltage
    gc.collect()


def plot_group_comparison(
    ax,
    wt_values: pd.Series,
    ko_values: pd.Series,
    ylabel: str = "",
    title: str = "",
    colors: dict | None = None,
) -> dict:
    """Plot WT vs KO comparison with Mann-Whitney statistics."""
    if colors is None:
        colors = {"WT": "#378ADD", "KO": "#D85A30"}

    stat, pval = mannwhitneyu(wt_values, ko_values, alternative="two-sided")
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

    for label, values in [("WT", wt_values), ("KO", ko_values)]:
        ax.scatter([label] * len(values), values,
                   color=colors[label], alpha=0.7, s=60, zorder=3)
        ax.errorbar(label, values.mean(), yerr=values.sem(),
                    fmt="_", markersize=28, markeredgewidth=2.5,
                    color=colors[label], capsize=5, capthick=2)

    ax.annotate(f"{sig}\np={pval:.3f}", xy=(0.5, 0.93),
                xycoords="axes fraction", ha="center", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return {"pval": pval, "sig": sig,
            "wt_mean": wt_values.mean(), "ko_mean": ko_values.mean(),
            "wt_n": len(wt_values), "ko_n": len(ko_values)}
