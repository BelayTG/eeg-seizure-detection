"""
preprocessing.py
----------------
EEG signal preprocessing utilities for seizure-associated discharge detection.

Handles:
    - Loading .abf files via pyabf
    - Artifact rejection (voltage range filtering)
    - Epoch extraction from continuous recordings
    - Baseline threshold estimation

Usage:
    from preprocessing import load_abf, remove_artifacts, extract_epoch, estimate_baseline
"""

import numpy as np
import pyabf


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_abf(file_path: str, channel: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load a single-channel EEG recording from an .abf file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .abf file.
    channel : int
        Channel index to load (default 0 = first channel).

    Returns
    -------
    time : np.ndarray
        Time array in seconds.
    voltage : np.ndarray
        Voltage array in mV.
    sampling_rate : float
        Sampling rate in Hz.

    Raises
    ------
    FileNotFoundError
        If the .abf file does not exist at the given path.
    """
    abf = pyabf.ABF(file_path)
    abf.setSweep(0, channel=channel)
    time = abf.sweepX.copy()
    voltage = abf.sweepY.copy()
    sampling_rate = abf.dataRate
    return time, voltage, sampling_rate


# ---------------------------------------------------------------------------
# Artifact rejection
# ---------------------------------------------------------------------------

def remove_artifacts(
    time: np.ndarray,
    voltage: np.ndarray,
    v_min: float = -10.0,
    v_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove samples outside a physiologically plausible voltage range.

    Artifact rejection is applied uniformly across genotypes so that
    comparisons between WT and KO groups are unbiased.

    Parameters
    ----------
    time : np.ndarray
        Time array in seconds.
    voltage : np.ndarray
        Voltage array in mV.
    v_min : float
        Lower bound for valid voltage (default -10 mV).
    v_max : float
        Upper bound for valid voltage (default +10 mV).

    Returns
    -------
    time_clean : np.ndarray
        Time array with artifact samples removed.
    voltage_clean : np.ndarray
        Voltage array with artifact samples removed.
    """
    valid = (voltage >= v_min) & (voltage <= v_max)
    return time[valid], voltage[valid]


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def extract_epoch(
    time: np.ndarray,
    voltage: np.ndarray,
    sampling_rate: float,
    start_min: float,
    duration_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a contiguous epoch from a continuous EEG recording.

    Parameters
    ----------
    time : np.ndarray
        Full time array in seconds.
    voltage : np.ndarray
        Full voltage array in mV.
    sampling_rate : float
        Sampling rate in Hz.
    start_min : float
        Epoch start time in minutes.
    duration_min : float
        Epoch duration in minutes.

    Returns
    -------
    epoch_time : np.ndarray
        Time array for the extracted epoch (seconds, relative to recording start).
    epoch_voltage : np.ndarray
        Voltage array for the extracted epoch (mV).

    Notes
    -----
    Indices are computed from start_min and duration_min to avoid
    floating-point drift when iterating over many epochs.
    """
    start_idx = int(start_min * 60 * sampling_rate)
    end_idx = int((start_min + duration_min) * 60 * sampling_rate)
    end_idx = min(end_idx, len(voltage))
    return time[start_idx:end_idx], voltage[start_idx:end_idx]


# ---------------------------------------------------------------------------
# Baseline estimation
# ---------------------------------------------------------------------------

def estimate_baseline(
    voltage: np.ndarray,
    sampling_rate: float,
    baseline_duration_s: float = 3600,
    coverage_threshold: float = 0.97,
    min_threshold_mv: float = 0.01,
    increment_mv: float = 0.0005,
) -> float:
    """
    Estimate a data-driven baseline activity threshold from the initial
    recording segment.

    The threshold is the smallest value T such that at least
    `coverage_threshold` fraction of the baseline window falls below T.
    This approach adapts to animal-to-animal variability in baseline
    noise without requiring manual inspection.

    Parameters
    ----------
    voltage : np.ndarray
        Full voltage array in mV (artifact-cleaned).
    sampling_rate : float
        Sampling rate in Hz.
    baseline_duration_s : float
        Duration of the baseline window in seconds (default 3600 = 1 hour).
    coverage_threshold : float
        Fraction of baseline samples that must fall below the threshold
        (default 0.97, i.e. 97th percentile approximation).
    min_threshold_mv : float
        Starting value for threshold search in mV (default 0.01 mV).
    increment_mv : float
        Step size for threshold search in mV (default 0.5 µV).

    Returns
    -------
    baseline : float
        Estimated baseline threshold in mV.
    """
    n_samples = int(baseline_duration_s * sampling_rate)
    baseline_window = voltage[:n_samples]

    threshold = min_threshold_mv
    while True:
        if np.sum(baseline_window < threshold) / len(baseline_window) >= coverage_threshold:
            return threshold
        threshold += increment_mv


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_group_from_manifest(
    manifest_df,
    data_dir: str,
    epoch_duration_min: float = 0.5,
    channel: int = 0,
    v_min: float = -10.0,
    v_max: float = 10.0,
) -> list[dict]:
    """
    Load and preprocess all epochs defined in a metadata manifest DataFrame.

    The manifest must contain columns:
        - 'File'        : filename of the .abf recording
        - 'Start_Times' : comma-separated epoch start times in minutes

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Metadata table loaded from Excel (one row per recording).
    data_dir : str
        Directory containing the .abf files.
    epoch_duration_min : float
        Duration of each epoch in minutes (default 0.5 = 30 seconds).
    channel : int
        ABF channel index to load (default 0).
    v_min, v_max : float
        Artifact rejection voltage bounds in mV.

    Returns
    -------
    epochs : list of dict
        Each dict contains:
            'file'          : source filename
            'start_min'     : epoch start time in minutes
            'time'          : np.ndarray — epoch time in seconds
            'voltage'       : np.ndarray — artifact-cleaned voltage in mV
            'sampling_rate' : float — Hz
            'baseline'      : float — estimated baseline threshold in mV
    """
    import os

    epochs = []
    for _, row in manifest_df.iterrows():
        file_path = os.path.join(data_dir, row["File"])
        start_times = [
            float(t.strip())
            for t in str(row["Start_Times"]).split(",")
            if t.strip()
        ]

        time_full, voltage_full, fs = load_abf(file_path, channel=channel)
        time_clean, voltage_clean = remove_artifacts(
            time_full, voltage_full, v_min=v_min, v_max=v_max
        )
        baseline = estimate_baseline(voltage_clean, fs)

        for start in start_times:
            t_epoch, v_epoch = extract_epoch(
                time_clean, voltage_clean, fs, start, epoch_duration_min
            )
            epochs.append(
                {
                    "file": row["File"],
                    "start_min": start,
                    "time": t_epoch,
                    "voltage": v_epoch,
                    "sampling_rate": fs,
                    "baseline": baseline,
                }
            )

    return epochs
