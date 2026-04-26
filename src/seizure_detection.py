"""
seizure_detection.py
--------------------
Comprehensive automated seizure analysis for kainic acid EEG recordings.

Analyses
--------
1. IED detection          — interictal epileptiform discharges
2. Seizure detection      — sustained ictal events (>5 seconds)
3. Seizure morphology     — onset frequency, amplitude evolution
4. CA3-cortex propagation — which region seizes first?
5. Post-ictal suppression — EEG suppression after each seizure
6. Spectral evolution     — time-frequency dynamics during seizures
7. Per-mouse metrics      — 12 summary measures per animal

Recording structure
-------------------
    Two consecutive 2h ABF files per mouse (flat directory).
    Files sorted alphabetically, paired sequentially.
    Channel 0 = CA3, Channel 1 = cortex.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import find_peaks, butter, filtfilt, welch, spectrogram
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------

def pair_files(data_dir: str) -> list[tuple[str, str]]:
    """
    Pair consecutive ABF files. Files sorted alphabetically,
    paired sequentially: [0,1]=mouse1, [2,3]=mouse2, etc.
    """
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".abf")])
    pairs = []
    for i in range(0, len(files) - 1, 2):
        pairs.append((files[i], files[i + 1]))
    if len(files) % 2 != 0:
        print(f"  Warning: {files[-1]} is unpaired — excluded")
    return pairs


def get_mouse_id(file1: str) -> str:
    """Mouse ID = first 7 characters of file1."""
    return file1[:7]


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------

def load_channel(file_path: str, channel: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a single channel from an ABF file."""
    abf = pyabf.ABF(file_path)
    abf.setSweep(0, channel=channel)
    time = abf.sweepX.copy()
    voltage = abf.sweepY.copy()
    fs = abf.dataRate
    del abf
    gc.collect()
    return time, voltage, fs


def concatenate_recordings(
    file1_path: str,
    file2_path: str,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Concatenate two 2h recordings into a single 4h trace."""
    time1, voltage1, fs = load_channel(file1_path, channel)
    time2, voltage2, _ = load_channel(file2_path, channel)
    time_offset = time1[-1] + 1.0 / fs
    time = np.concatenate([time1, time2 + time_offset])
    voltage = np.concatenate([voltage1, voltage2])
    del time1, voltage1, time2, voltage2
    gc.collect()
    return time, voltage, fs


def load_dual_channel(
    file1_path: str,
    file2_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and concatenate both CA3 and cortex channels."""
    time, ca3, fs = concatenate_recordings(file1_path, file2_path, channel=0)
    _, ctx, _ = concatenate_recordings(file1_path, file2_path, channel=1)
    return time, ca3, ctx, fs


def remove_artifacts(time, voltage, v_min=-10.0, v_max=10.0):
    valid = (voltage >= v_min) & (voltage <= v_max)
    return time[valid], voltage[valid]


def estimate_baseline(voltage, fs, duration_s=3600, coverage=0.97):
    n = int(min(duration_s * fs, len(voltage)))
    window = voltage[:n]
    threshold = 0.01
    while True:
        if np.sum(window < threshold) / len(window) >= coverage:
            return float(threshold)
        threshold += 0.0005


# ---------------------------------------------------------------------------
# IED detection
# ---------------------------------------------------------------------------

def detect_ieds(
    time, voltage, fs,
    lower_threshold,
    upper_threshold=1.5,
    min_prominence=0.2,
    max_width_s=0.2,
    min_interval_s=0.1,
) -> pd.DataFrame:
    """Detect interictal epileptiform discharges."""
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
    time, voltage, fs, ieds,
    min_seizure_duration_s=5.0,
    max_ied_gap_s=2.0,
    min_ieds_per_seizure=5,
) -> pd.DataFrame:
    """
    Detect sustained seizures by clustering dense IED activity.
    A seizure = cluster of IEDs with gap < max_ied_gap_s,
    duration > min_seizure_duration_s, count >= min_ieds_per_seizure.
    """
    if len(ieds) < min_ieds_per_seizure:
        return pd.DataFrame(columns=[
            "onset_s", "offset_s", "duration_s", "n_ieds", "mean_amplitude_mV"
        ])

    ied_times = ieds["time_s"].values
    seizures = []
    cluster = [ied_times[0]]

    for i in range(1, len(ied_times)):
        if ied_times[i] - ied_times[i-1] <= max_ied_gap_s:
            cluster.append(ied_times[i])
        else:
            duration = cluster[-1] - cluster[0]
            if duration >= min_seizure_duration_s and len(cluster) >= min_ieds_per_seizure:
                c_ieds = ieds[(ieds["time_s"] >= cluster[0]) & (ieds["time_s"] <= cluster[-1])]
                seizures.append({
                    "onset_s": cluster[0],
                    "offset_s": cluster[-1],
                    "duration_s": duration,
                    "n_ieds": len(cluster),
                    "mean_amplitude_mV": c_ieds["voltage_mV"].mean(),
                })
            cluster = [ied_times[i]]

    # Last cluster
    duration = cluster[-1] - cluster[0]
    if duration >= min_seizure_duration_s and len(cluster) >= min_ieds_per_seizure:
        c_ieds = ieds[(ieds["time_s"] >= cluster[0]) & (ieds["time_s"] <= cluster[-1])]
        seizures.append({
            "onset_s": cluster[0], "offset_s": cluster[-1],
            "duration_s": duration, "n_ieds": len(cluster),
            "mean_amplitude_mV": c_ieds["voltage_mV"].mean(),
        })

    return pd.DataFrame(seizures) if seizures else pd.DataFrame(
        columns=["onset_s", "offset_s", "duration_s", "n_ieds", "mean_amplitude_mV"]
    )


# ---------------------------------------------------------------------------
# Advanced Analysis 1: Seizure morphology
# ---------------------------------------------------------------------------

def analyze_seizure_morphology(
    time: np.ndarray,
    voltage: np.ndarray,
    fs: float,
    seizures: pd.DataFrame,
    n_segments: int = 5,
) -> pd.DataFrame:
    """
    Analyze spectral and amplitude characteristics of each seizure.

    For each seizure, divides the event into n_segments and computes:
    - Dominant frequency at onset, middle, and offset
    - Peak amplitude at each segment (amplitude evolution)
    - Whether seizure crescendos or decrescendos

    Parameters
    ----------
    n_segments : int — number of time segments to divide each seizure into

    Returns
    -------
    pd.DataFrame — one row per seizure with morphology metrics
    """
    rows = []
    for _, szr in seizures.iterrows():
        onset_idx = int(szr["onset_s"] * fs)
        offset_idx = int(szr["offset_s"] * fs)
        seizure_signal = voltage[onset_idx:offset_idx]

        if len(seizure_signal) < int(fs):
            continue

        # Segment analysis
        seg_len = len(seizure_signal) // n_segments
        seg_freqs = []
        seg_amps = []

        for seg in range(n_segments):
            seg_signal = seizure_signal[seg * seg_len:(seg + 1) * seg_len]
            if len(seg_signal) < 10:
                continue
            freqs, psd = welch(seg_signal, fs=fs, nperseg=min(int(fs), len(seg_signal)))
            dom_freq = freqs[np.argmax(psd)]
            seg_freqs.append(float(dom_freq))
            seg_amps.append(float(np.max(np.abs(seg_signal))))

        if not seg_freqs:
            continue

        # Amplitude evolution — crescendo vs decrescendo
        amp_slope = float(np.polyfit(range(len(seg_amps)), seg_amps, 1)[0])

        rows.append({
            "onset_s": szr["onset_s"],
            "duration_s": szr["duration_s"],
            "onset_freq_hz": seg_freqs[0],
            "peak_freq_hz": max(seg_freqs),
            "offset_freq_hz": seg_freqs[-1],
            "mean_freq_hz": float(np.mean(seg_freqs)),
            "onset_amplitude_mV": seg_amps[0],
            "peak_amplitude_mV": max(seg_amps),
            "amplitude_slope": amp_slope,
            "pattern": "crescendo" if amp_slope > 0 else "decrescendo",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Advanced Analysis 2: CA3-cortex propagation
# ---------------------------------------------------------------------------

def analyze_propagation(
    time: np.ndarray,
    ca3: np.ndarray,
    ctx: np.ndarray,
    fs: float,
    seizures_ca3: pd.DataFrame,
    seizures_ctx: pd.DataFrame,
    tolerance_s: float = 5.0,
) -> pd.DataFrame:
    """
    Determine whether seizures onset in CA3 or cortex first.

    Matches seizures detected in each channel and computes the
    time difference (positive = CA3 leads, negative = cortex leads).

    Parameters
    ----------
    seizures_ca3 : pd.DataFrame — seizures from CA3 channel
    seizures_ctx : pd.DataFrame — seizures from cortex channel
    tolerance_s : float — maximum time difference to match seizures (default 5s)

    Returns
    -------
    pd.DataFrame — matched seizure pairs with propagation delay
    """
    if len(seizures_ca3) == 0 or len(seizures_ctx) == 0:
        return pd.DataFrame(columns=[
            "ca3_onset_s", "ctx_onset_s", "delay_s", "leader"
        ])

    rows = []
    for _, szr_ca3 in seizures_ca3.iterrows():
        # Find matching cortex seizure within tolerance
        candidates = seizures_ctx[
            np.abs(seizures_ctx["onset_s"] - szr_ca3["onset_s"]) <= tolerance_s
        ]
        if len(candidates) == 0:
            continue

        closest = candidates.iloc[
            np.argmin(np.abs(candidates["onset_s"] - szr_ca3["onset_s"]))
        ]
        delay = szr_ca3["onset_s"] - closest["onset_s"]
        rows.append({
            "ca3_onset_s": szr_ca3["onset_s"],
            "ctx_onset_s": closest["onset_s"],
            "delay_s": delay,
            "leader": "CA3" if delay < 0 else "Cortex" if delay > 0 else "Simultaneous",
            "ca3_duration_s": szr_ca3["duration_s"],
            "ctx_duration_s": closest["duration_s"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Advanced Analysis 3: Post-ictal suppression
# ---------------------------------------------------------------------------

def detect_postictal_suppression(
    time: np.ndarray,
    voltage: np.ndarray,
    fs: float,
    seizures: pd.DataFrame,
    window_s: float = 60.0,
    suppression_threshold_factor: float = 0.3,
) -> pd.DataFrame:
    """
    Detect post-ictal EEG suppression after each seizure.

    Suppression is defined as a period where signal RMS falls below
    suppression_threshold_factor × pre-seizure RMS.

    Parameters
    ----------
    window_s : float — post-ictal window to analyze (default 60s)
    suppression_threshold_factor : float — RMS threshold as fraction of baseline

    Returns
    -------
    pd.DataFrame — suppression duration per seizure
    """
    if len(seizures) == 0:
        return pd.DataFrame(columns=[
            "seizure_onset_s", "suppression_duration_s", "suppression_detected"
        ])

    rows = []
    epoch_s = 2.0  # 2-second epochs for RMS computation

    for _, szr in seizures.iterrows():
        offset_idx = int(szr["offset_s"] * fs)
        pre_start = max(0, int((szr["onset_s"] - 30) * fs))
        pre_end = int(szr["onset_s"] * fs)

        if pre_end <= pre_start:
            continue

        # Pre-seizure RMS
        pre_signal = voltage[pre_start:pre_end]
        pre_rms = float(np.sqrt(np.mean(pre_signal ** 2)))
        if pre_rms == 0:
            continue

        threshold = suppression_threshold_factor * pre_rms

        # Post-seizure RMS in 2s epochs
        post_end = min(len(voltage), offset_idx + int(window_s * fs))
        post_signal = voltage[offset_idx:post_end]

        suppression_dur = 0.0
        n_epochs = len(post_signal) // int(epoch_s * fs)

        for ep in range(n_epochs):
            ep_signal = post_signal[ep * int(epoch_s * fs):(ep + 1) * int(epoch_s * fs)]
            ep_rms = float(np.sqrt(np.mean(ep_signal ** 2)))
            if ep_rms < threshold:
                suppression_dur += epoch_s
            else:
                break  # Suppression ends at first recovery epoch

        rows.append({
            "seizure_onset_s": szr["onset_s"],
            "seizure_duration_s": szr["duration_s"],
            "suppression_duration_s": suppression_dur,
            "suppression_detected": suppression_dur > 0,
            "pre_seizure_rms": pre_rms,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Advanced Analysis 4: Spectral evolution during seizures
# ---------------------------------------------------------------------------

def compute_seizure_spectrogram(
    voltage: np.ndarray,
    fs: float,
    onset_s: float,
    duration_s: float,
    pre_s: float = 10.0,
    post_s: float = 10.0,
    freq_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram for a seizure event including pre and post periods.

    Parameters
    ----------
    voltage : full voltage array
    fs : sampling rate
    onset_s : seizure onset in seconds
    duration_s : seizure duration in seconds
    pre_s : seconds before onset to include
    post_s : seconds after offset to include
    freq_max : maximum frequency to display

    Returns
    -------
    frequencies, times, Sxx (power spectrogram)
    """
    start_idx = max(0, int((onset_s - pre_s) * fs))
    end_idx = min(len(voltage), int((onset_s + duration_s + post_s) * fs))
    signal = voltage[start_idx:end_idx]

    nperseg = min(int(fs), len(signal) // 4)
    noverlap = nperseg // 2

    freqs, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = freqs <= freq_max

    return freqs[mask], times, Sxx[mask, :]


# ---------------------------------------------------------------------------
# Per-mouse full pipeline
# ---------------------------------------------------------------------------

def process_mouse(
    file1_path: str,
    file2_path: str,
    mouse_id: str,
    channel_ca3: int = 0,
    channel_ctx: int = 1,
    run_advanced: bool = True,
) -> dict:
    """
    Run full detection and analysis pipeline on one mouse.

    Parameters
    ----------
    run_advanced : bool — run morphology, propagation, suppression analyses

    Returns
    -------
    dict with all metrics and sub-DataFrames
    """
    # Load dual channel
    time, ca3, ctx, fs = load_dual_channel(file1_path, file2_path)

    # Process CA3
    time_clean, ca3_clean = remove_artifacts(time, ca3)
    baseline = estimate_baseline(ca3_clean, fs)
    lower_threshold = 2.0 * baseline

    ieds = detect_ieds(time_clean, ca3_clean, fs, lower_threshold)
    seizures = detect_seizures(time_clean, ca3_clean, fs, ieds)

    # Process cortex for propagation
    _, ctx_clean = remove_artifacts(time, ctx)
    baseline_ctx = estimate_baseline(ctx_clean, fs)
    ieds_ctx = detect_ieds(time_clean, ctx_clean, fs, 2.0 * baseline_ctx)
    seizures_ctx = detect_seizures(time_clean, ctx_clean, fs, ieds_ctx)

    recording_duration_s = len(time_clean) / fs
    duration_min = recording_duration_s / 60.0

    # Core metrics
    n_ieds = len(ieds)
    ied_rate = n_ieds / duration_min if duration_min > 0 else 0.0
    mean_ied_interval = float(np.diff(ieds["time_s"].values).mean()) if n_ieds > 1 else 0.0
    n_seizures = len(seizures)
    total_szr_dur = float(seizures["duration_s"].sum()) if n_seizures > 0 else 0.0
    mean_szr_dur = float(seizures["duration_s"].mean()) if n_seizures > 0 else 0.0
    seizure_burden = 100.0 * total_szr_dur / recording_duration_s
    first_seizure_min = float(seizures["onset_s"].min()) / 60.0 if n_seizures > 0 else np.nan

    result = {
        "mouse_id": mouse_id,
        "n_ieds": n_ieds,
        "ied_rate_per_min": round(ied_rate, 4),
        "mean_ied_interval_s": round(mean_ied_interval, 4),
        "n_seizures": n_seizures,
        "total_seizure_dur_s": round(total_szr_dur, 2),
        "mean_seizure_dur_s": round(mean_szr_dur, 2),
        "seizure_burden_pct": round(seizure_burden, 4),
        "first_seizure_min": round(first_seizure_min, 2) if not np.isnan(first_seizure_min) else np.nan,
        "recording_duration_min": round(duration_min, 2),
        "baseline_mV": round(baseline, 6),
        "ieds": ieds,
        "seizures": seizures,
        "seizures_ctx": seizures_ctx,
    }

    if run_advanced and n_seizures > 0:
        # Morphology
        morphology = analyze_seizure_morphology(time_clean, ca3_clean, fs, seizures)
        result["morphology"] = morphology
        if len(morphology) > 0:
            result["mean_onset_freq_hz"] = round(morphology["onset_freq_hz"].mean(), 2)
            result["pct_crescendo"] = round(
                100 * (morphology["pattern"] == "crescendo").mean(), 1
            )
        else:
            result["mean_onset_freq_hz"] = np.nan
            result["pct_crescendo"] = np.nan

        # Propagation
        propagation = analyze_propagation(
            time_clean, ca3_clean, ctx_clean, fs, seizures, seizures_ctx
        )
        result["propagation"] = propagation
        if len(propagation) > 0:
            result["mean_propagation_delay_s"] = round(propagation["delay_s"].mean(), 3)
            result["pct_ca3_leads"] = round(
                100 * (propagation["leader"] == "CA3").mean(), 1
            )
        else:
            result["mean_propagation_delay_s"] = np.nan
            result["pct_ca3_leads"] = np.nan

        # Post-ictal suppression
        suppression = detect_postictal_suppression(time_clean, ca3_clean, fs, seizures)
        result["suppression"] = suppression
        if len(suppression) > 0:
            result["mean_suppression_dur_s"] = round(
                suppression["suppression_duration_s"].mean(), 2
            )
            result["pct_with_suppression"] = round(
                100 * suppression["suppression_detected"].mean(), 1
            )
        else:
            result["mean_suppression_dur_s"] = np.nan
            result["pct_with_suppression"] = np.nan
    else:
        result["morphology"] = pd.DataFrame()
        result["propagation"] = pd.DataFrame()
        result["suppression"] = pd.DataFrame()
        result["mean_onset_freq_hz"] = np.nan
        result["pct_crescendo"] = np.nan
        result["mean_propagation_delay_s"] = np.nan
        result["pct_ca3_leads"] = np.nan
        result["mean_suppression_dur_s"] = np.nan
        result["pct_with_suppression"] = np.nan

    del time, ca3, ctx, time_clean, ca3_clean, ctx_clean
    gc.collect()
    return result


def process_group(
    data_dir: str,
    group: str = "WT",
    run_advanced: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Process all mice in a group directory."""
    pairs = pair_files(data_dir)
    if verbose:
        print(f"  {len(pairs)} mice in {group}")

    metric_cols = [
        "mouse_id", "group", "n_ieds", "ied_rate_per_min", "mean_ied_interval_s",
        "n_seizures", "total_seizure_dur_s", "mean_seizure_dur_s",
        "seizure_burden_pct", "first_seizure_min", "recording_duration_min",
        "mean_onset_freq_hz", "pct_crescendo",
        "mean_propagation_delay_s", "pct_ca3_leads",
        "mean_suppression_dur_s", "pct_with_suppression",
    ]

    rows = []
    details = {}

    for i, (file1, file2) in enumerate(pairs):
        mouse_id = get_mouse_id(file1)
        f1 = os.path.join(data_dir, file1)
        f2 = os.path.join(data_dir, file2)

        if verbose:
            print(f"  Mouse {i+1}/{len(pairs)}: {mouse_id}")

        try:
            result = process_mouse(f1, f2, mouse_id, run_advanced=run_advanced)
            details[mouse_id] = {
                "ieds": result.pop("ieds"),
                "seizures": result.pop("seizures"),
                "seizures_ctx": result.pop("seizures_ctx"),
                "morphology": result.pop("morphology"),
                "propagation": result.pop("propagation"),
                "suppression": result.pop("suppression"),
            }
            result["group"] = group
            rows.append({col: result.get(col, np.nan) for col in metric_cols})

            if verbose:
                print(f"    IEDs:{result['n_ieds']} | Seizures:{result['n_seizures']} | "
                      f"Burden:{result['seizure_burden_pct']:.1f}% | "
                      f"CA3 leads:{result['pct_ca3_leads']}%")
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")

    return pd.DataFrame(rows), details


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eeg_with_events(
    file1_path, file2_path, ieds, seizures, mouse_id,
    plot_duration_min=30.0, save_path=None
):
    """Plot EEG with IEDs and seizures overlaid."""
    time, voltage, fs = concatenate_recordings(file1_path, file2_path, channel=0)
    time, voltage = remove_artifacts(time, voltage)

    end_idx = int(plot_duration_min * 60 * fs)
    t_plot = time[:end_idx]
    v_plot = voltage[:end_idx]
    ieds_plot = ieds[ieds["time_s"] <= t_plot[-1]]
    szr_plot = seizures[seizures["onset_s"] <= t_plot[-1]]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_plot / 60, v_plot, color="#2C2C2A", lw=0.3, label="EEG (CA3)")
    if len(ieds_plot) > 0:
        ax.scatter(ieds_plot["time_s"] / 60, ieds_plot["voltage_mV"],
                   color="#D85A30", s=12, zorder=5,
                   label=f"IEDs (n={len(ieds_plot)} shown)")
    for _, szr in szr_plot.iterrows():
        ax.axvspan(szr["onset_s"] / 60, szr["offset_s"] / 60,
                   color="#59A14F", alpha=0.25)
    if len(szr_plot) > 0:
        ax.axvspan(0, 0, color="#59A14F", alpha=0.4,
                   label=f"Seizures (n={len(szr_plot)} shown)")
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


def plot_seizure_spectrogram(
    file1_path, file2_path, seizures, mouse_id,
    save_path=None
):
    """Plot spectrogram of the first detected seizure."""
    if len(seizures) == 0:
        print(f"No seizures detected for {mouse_id}")
        return

    time, voltage, fs = concatenate_recordings(file1_path, file2_path, channel=0)
    _, voltage = remove_artifacts(time, voltage)

    szr = seizures.iloc[0]
    freqs, times, Sxx = compute_seizure_spectrogram(
        voltage, fs, szr["onset_s"], szr["duration_s"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # Raw EEG
    pre_s, post_s = 10.0, 10.0
    start_idx = max(0, int((szr["onset_s"] - pre_s) * fs))
    end_idx = min(len(voltage), int((szr["onset_s"] + szr["duration_s"] + post_s) * fs))
    t_seg = np.arange(end_idx - start_idx) / fs - pre_s
    v_seg = voltage[start_idx:end_idx]

    axes[0].plot(t_seg, v_seg, color="#2C2C2A", lw=0.5)
    axes[0].axvline(0, color="#D85A30", lw=1.5, ls="--", label="Seizure onset")
    axes[0].axvline(szr["duration_s"], color="#378ADD", lw=1.5, ls="--", label="Seizure offset")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(f"Seizure spectrogram — Mouse {mouse_id}")
    axes[0].legend(fontsize=9)

    # Spectrogram
    t_spec = times - pre_s
    im = axes[1].pcolormesh(t_spec, freqs, 10 * np.log10(Sxx + 1e-12),
                             cmap="hot_r", shading="auto")
    plt.colorbar(im, ax=axes[1], label="Power (dB)")
    axes[1].axvline(0, color="white", lw=1.5, ls="--")
    axes[1].axvline(szr["duration_s"], color="cyan", lw=1.5, ls="--")
    axes[1].set_xlabel("Time relative to seizure onset (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    del time, voltage
    gc.collect()


def plot_group_comparison(ax, wt_values, ko_values, ylabel="", title="",
                          colors=None):
    """Plot WT vs KO with Mann-Whitney statistics."""
    if colors is None:
        colors = {"WT": "#378ADD", "KO": "#D85A30"}
    wt_clean = wt_values.dropna()
    ko_clean = ko_values.dropna()
    if len(wt_clean) < 2 or len(ko_clean) < 2:
        ax.set_visible(False)
        return {}
    stat, pval = mannwhitneyu(wt_clean, ko_clean, alternative="two-sided")
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    for label, values in [("WT", wt_clean), ("KO", ko_clean)]:
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
            "wt_mean": wt_clean.mean(), "ko_mean": ko_clean.mean()}
