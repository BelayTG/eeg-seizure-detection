"""
detection.py
------------
Seizure-associated epileptiform discharge (SAD) detection from EEG recordings.

Detection is based on four physiologically motivated criteria:
    1. Amplitude threshold  : peak must exceed 2x the adaptive baseline
    2. Prominence           : peak must stand out from local signal (min 0.2 mV)
    3. Width                : sharp transients only (max 200 ms)
    4. Refractory period    : minimum 100 ms between consecutive events

These parameters are applied uniformly across WT and KO groups to ensure
unbiased comparison of network hyperexcitability.

Typical usage
-------------
    from preprocessing import load_abf, remove_artifacts, estimate_baseline
    from detection import detect_discharges, compute_psd, summarise_detections

    time, voltage, fs = load_abf("recording.abf")
    time, voltage = remove_artifacts(time, voltage)
    baseline = estimate_baseline(voltage, fs)

    events = detect_discharges(time, voltage, fs, lower_threshold=2 * baseline)
    psd_freq, psd_power = compute_psd(voltage, fs)
    summary = summarise_detections(events, recording_duration_s=len(time) / fs)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_discharges(
    time: np.ndarray,
    voltage: np.ndarray,
    sampling_rate: float,
    lower_threshold: float,
    upper_threshold: float = 1.5,
    min_prominence: float = 0.2,
    max_width_s: float = 0.2,
    min_interval_s: float = 0.1,
) -> pd.DataFrame:
    """
    Detect seizure-associated epileptiform discharges (SADs) in an EEG trace.

    Uses scipy.signal.find_peaks with amplitude, prominence, width, and
    refractory period constraints. All thresholds are in physical units
    (mV, seconds) rather than sample counts, making the function
    sampling-rate agnostic.

    Parameters
    ----------
    time : np.ndarray
        Time array in seconds.
    voltage : np.ndarray
        Voltage array in mV (artifact-cleaned).
    sampling_rate : float
        Sampling rate in Hz.
    lower_threshold : float
        Minimum peak amplitude in mV. Typically set to 2 × baseline.
    upper_threshold : float
        Maximum peak amplitude in mV (default 1.5 mV).
        Caps at physiological limit to exclude residual movement artifacts.
    min_prominence : float
        Minimum peak prominence in mV (default 0.2 mV).
        Ensures detected peaks stand above local signal fluctuations.
    max_width_s : float
        Maximum peak width in seconds (default 0.2 s = 200 ms).
        Restricts detection to sharp epileptiform transients.
    min_interval_s : float
        Minimum interval between consecutive events in seconds (default 0.1 s).
        Implements a physiological refractory period.

    Returns
    -------
    events : pd.DataFrame
        One row per detected discharge, columns:
            'time_s'      : discharge time in seconds
            'voltage_mV'  : discharge peak voltage in mV
            'prominence'  : peak prominence in mV

    Notes
    -----
    Detected events reflect seizure-associated network hyperexcitability
    and are not restricted to precisely defined ictal epochs. Event counts
    serve as a relative measure for WT vs KO comparison.
    """
    max_width_samples = max_width_s * sampling_rate
    min_distance_samples = int(min_interval_s * sampling_rate)

    peaks, properties = find_peaks(
        voltage,
        height=(lower_threshold, upper_threshold),
        prominence=min_prominence,
        width=(None, max_width_samples),
        distance=min_distance_samples,
    )

    if len(peaks) == 0:
        return pd.DataFrame(columns=["time_s", "voltage_mV", "prominence"])

    events = pd.DataFrame(
        {
            "time_s": time[peaks],
            "voltage_mV": voltage[peaks],
            "prominence": properties["prominences"],
        }
    )
    return events


# ---------------------------------------------------------------------------
# Power spectral density
# ---------------------------------------------------------------------------

def compute_psd(
    voltage: np.ndarray,
    sampling_rate: float,
    freq_max: float = 50.0,
    nperseg: int = 2048,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectral density (PSD) of an EEG epoch using
    Welch's method.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage array in mV.
    sampling_rate : float
        Sampling rate in Hz.
    freq_max : float
        Upper frequency limit in Hz (default 50 Hz covers standard EEG bands).
    nperseg : int
        Samples per Welch segment (default 2048).
        Larger values give finer frequency resolution.
    normalize : bool
        If True, normalize PSD to unit area so recordings with different
        absolute power can be directly compared (default True).

    Returns
    -------
    frequencies : np.ndarray
        Frequency array in Hz (0 to freq_max).
    psd : np.ndarray
        Power spectral density. Units are mV²/Hz if normalize=False,
        or dimensionless (area = 1) if normalize=True.
    """
    frequencies, psd = welch(voltage, fs=sampling_rate, nperseg=nperseg)

    mask = frequencies <= freq_max
    frequencies = frequencies[mask]
    psd = psd[mask]

    if normalize:
        area = np.trapz(psd, frequencies)
        if area > 0:
            psd = psd / area

    return frequencies, psd


# ---------------------------------------------------------------------------
# Band power
# ---------------------------------------------------------------------------

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def compute_band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    bands: dict | None = None,
) -> dict[str, float]:
    """
    Compute relative power in standard EEG frequency bands.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz (from compute_psd).
    psd : np.ndarray
        Power spectral density array (from compute_psd).
    bands : dict, optional
        Dictionary mapping band name to (low_hz, high_hz) tuple.
        Defaults to delta, theta, alpha, beta, gamma.

    Returns
    -------
    band_powers : dict
        Band name → integrated power (mV²/Hz or relative if PSD is normalized).

    Example
    -------
    >>> freqs, psd = compute_psd(voltage, fs)
    >>> powers = compute_band_power(freqs, psd)
    >>> print(powers["gamma"])  # elevated gamma = seizure signature
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    band_powers = {}
    for band_name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() > 1:
            band_powers[band_name] = float(np.trapz(psd[mask], frequencies[mask]))
        else:
            band_powers[band_name] = 0.0

    return band_powers


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_detections(
    events: pd.DataFrame,
    recording_duration_s: float,
) -> dict:
    """
    Compute summary statistics for detected discharges in one recording.

    Parameters
    ----------
    events : pd.DataFrame
        Output of detect_discharges().
    recording_duration_s : float
        Total duration of the cleaned recording in seconds.

    Returns
    -------
    summary : dict
        Keys:
            'n_events'          : total discharge count
            'rate_per_min'      : discharge rate (events / minute)
            'mean_voltage_mV'   : mean peak voltage across events
            'std_voltage_mV'    : standard deviation of peak voltage
            'mean_prominence'   : mean peak prominence
            'recording_duration_s' : recording duration passed in
    """
    n = len(events)
    duration_min = recording_duration_s / 60.0

    summary = {
        "n_events": n,
        "rate_per_min": round(n / duration_min, 4) if duration_min > 0 else 0.0,
        "mean_voltage_mV": round(events["voltage_mV"].mean(), 4) if n > 0 else 0.0,
        "std_voltage_mV": round(events["voltage_mV"].std(), 4) if n > 0 else 0.0,
        "mean_prominence": round(events["prominence"].mean(), 4) if n > 0 else 0.0,
        "recording_duration_s": recording_duration_s,
    }
    return summary


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_group(
    epochs: list[dict],
    upper_threshold: float = 1.5,
    min_prominence: float = 0.2,
    max_width_s: float = 0.2,
    min_interval_s: float = 0.1,
) -> pd.DataFrame:
    """
    Run detection across all epochs returned by preprocessing.load_group_from_manifest.

    Parameters
    ----------
    epochs : list of dict
        Each dict must contain keys: 'file', 'start_min', 'time', 'voltage',
        'sampling_rate', 'baseline' — as returned by load_group_from_manifest.
    upper_threshold, min_prominence, max_width_s, min_interval_s
        Detection parameters passed through to detect_discharges().

    Returns
    -------
    results : pd.DataFrame
        One row per epoch, columns:
            'file', 'start_min', 'n_events', 'rate_per_min',
            'mean_voltage_mV', 'std_voltage_mV', 'mean_prominence',
            'recording_duration_s'
    """
    rows = []
    for ep in epochs:
        lower_threshold = 2.0 * ep["baseline"]
        events = detect_discharges(
            ep["time"],
            ep["voltage"],
            ep["sampling_rate"],
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            min_prominence=min_prominence,
            max_width_s=max_width_s,
            min_interval_s=min_interval_s,
        )
        summary = summarise_detections(events, recording_duration_s=len(ep["time"]) / ep["sampling_rate"])
        rows.append({"file": ep["file"], "start_min": ep["start_min"], **summary})

    return pd.DataFrame(rows)
