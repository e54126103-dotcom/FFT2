"""
Reusable FFT-based active noise cancellation analysis.

Command-line use:

    python analyze_noise.py

Streamlit use:

    streamlit run app.py

The main experiment reconstructs noise from selected complex FFT coefficients.
This preserves frequency, amplitude, and phase. The ideal anti-noise case is
included only as a reference and is not used as the main experiment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks


INPUT_WAV = Path("input_noise.wav")
OUTPUT_DIR = Path("output")
DEFAULT_TOP_N_VALUES = [1, 3, 5, 10, 20, 50]
DEFAULT_DELAYS_MS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_BURST_WINDOW_MS = 20.0
DEFAULT_BURST_PADDING_MS = 30.0
DEFAULT_BURST_THRESHOLD_MULTIPLIER = 4.0
DEFAULT_BURST_ATTENUATION = 0.2
MODE_HARMONIC = "基頻與倍頻模式"
MODE_ENERGY = "能量比例保留模式"
MODE_PEAK_BAND = "Peak 頻帶保留模式"
MODE_TOP_N = "Top N 頻率點模式（簡化比較用）"
RECONSTRUCTION_MODES = [MODE_HARMONIC, MODE_ENERGY, MODE_PEAK_BAND, MODE_TOP_N]
EPS = 1e-20


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono float64 and normalize to [-1, 1]."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_abs_for_dtype = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / max_abs_for_dtype
    else:
        audio = audio.astype(np.float64)

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def load_audio_mono(path: Path) -> tuple[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Put a WAV noise recording in this folder.")
    sample_rate, audio = wavfile.read(path)
    return sample_rate, normalize_audio_array(audio)


def safe_normalize_for_wav(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak > 1.0:
        return signal / peak
    return signal


def write_wav(path: str | Path, sample_rate: int, signal: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, safe_normalize_for_wav(signal).astype(np.float32))
    return path


def compute_metrics(original: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    residual_energy = float(np.sum(residual**2))
    original_energy = float(np.sum(original**2))
    return {
        "MSE": float(np.mean(residual**2)),
        "RMS_original": float(np.sqrt(np.mean(original**2))),
        "RMS_residual": float(np.sqrt(np.mean(residual**2))),
        "NRR_dB": 10.0 * math.log10((original_energy + EPS) / (residual_energy + EPS)),
    }


def frequency_mask(freqs: np.ndarray, min_freq_hz: float, max_freq_hz: float) -> np.ndarray:
    return (freqs >= min_freq_hz) & (freqs <= max_freq_hz)


def bins_in_band(freqs: np.ndarray, center_hz: float, bandwidth_hz: float) -> np.ndarray:
    half_width = bandwidth_hz
    return np.flatnonzero((freqs >= center_hz - half_width) & (freqs <= center_hz + half_width))


def selected_fft_from_bins(fft_values: np.ndarray, selected_bins: np.ndarray) -> np.ndarray:
    selected_fft = np.zeros_like(fft_values)
    selected_fft[np.unique(selected_bins)] = fft_values[np.unique(selected_bins)]
    return selected_fft


def reconstruct_from_bins(fft_values: np.ndarray, selected_bins: np.ndarray, signal_length: int) -> np.ndarray:
    """Reconstruct using selected complex FFT coefficients via rfft/irfft."""
    return np.fft.irfft(selected_fft_from_bins(fft_values, selected_bins), n=signal_length)


def zero_padded_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Delay a signal using zero padding, not circular shift."""
    if delay_samples <= 0:
        return signal.copy()
    delayed = np.zeros_like(signal)
    if delay_samples < signal.size:
        delayed[delay_samples:] = signal[:-delay_samples]
    return delayed


def samples_from_ms(sample_rate: int, duration_ms: float) -> int:
    return max(1, int(round(sample_rate * duration_ms / 1000.0)))


def moving_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    power = np.convolve(signal**2, kernel, mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def pad_boolean_mask(mask: np.ndarray, padding_samples: int) -> np.ndarray:
    if padding_samples <= 0 or not np.any(mask):
        return mask.copy()
    kernel = np.ones(2 * padding_samples + 1, dtype=np.int8)
    return np.convolve(mask.astype(np.int8), kernel, mode="same") > 0


def mask_to_segments(mask: np.ndarray, sample_rate: int) -> pd.DataFrame:
    if not np.any(mask):
        return pd.DataFrame(
            columns=["burst_id", "start_time_s", "end_time_s", "duration_ms", "start_sample", "end_sample"]
        )

    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    rows = []
    for burst_id, (start, end) in enumerate(zip(starts, ends), start=1):
        rows.append(
            {
                "burst_id": burst_id,
                "start_time_s": start / sample_rate,
                "end_time_s": end / sample_rate,
                "duration_ms": (end - start) / sample_rate * 1000.0,
                "start_sample": int(start),
                "end_sample": int(end),
            }
        )
    return pd.DataFrame(rows)


def find_peak_bins(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    min_freq_hz: float,
    max_freq_hz: float,
    max_peaks: int | None = None,
) -> np.ndarray:
    magnitude = np.abs(fft_values)
    search_indices = np.flatnonzero(frequency_mask(freqs, min_freq_hz, max_freq_hz))
    if search_indices.size == 0:
        return np.array([], dtype=int)

    search_magnitude = magnitude[search_indices]
    positive = search_magnitude[search_magnitude > 0]
    if positive.size == 0:
        return np.array([], dtype=int)

    prominence = max(float(np.max(search_magnitude)) * 0.005, float(np.median(positive)))
    peaks, _ = find_peaks(search_magnitude, prominence=prominence)
    peak_bins = search_indices[peaks]
    if peak_bins.size == 0:
        peak_bins = search_indices

    ranked = peak_bins[np.argsort(magnitude[peak_bins])[::-1]]
    if max_peaks is not None:
        ranked = ranked[:max_peaks]
    return ranked


def detect_fundamental_frequency(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    min_freq_hz: float,
    fundamental_max_freq_hz: float,
) -> dict[str, Any]:
    """Detect the strongest spectral peak in the fundamental search range."""
    peak_bins = find_peak_bins(fft_values, freqs, min_freq_hz, fundamental_max_freq_hz)
    if peak_bins.size == 0:
        raise ValueError("No fundamental peak was found in the selected search range.")
    f0_bin = int(peak_bins[0])
    return {
        "bin": f0_bin,
        "detected_f0_Hz": float(freqs[f0_bin]),
        "magnitude": float(np.abs(fft_values[f0_bin])),
        "phase_radians": float(np.angle(fft_values[f0_bin])),
    }


def build_harmonic_components(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    f0_hz: float,
    max_freq_hz: float,
    harmonic_bandwidth_hz: float,
    max_harmonics: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    power = np.abs(fft_values) ** 2
    total_energy = float(np.sum(power) + EPS)
    selected_bins: list[int] = []
    rows = []

    order = 1
    while order * f0_hz <= max_freq_hz:
        if max_harmonics is not None and order > max_harmonics:
            break

        target = order * f0_hz
        band_bins = bins_in_band(freqs, target, harmonic_bandwidth_hz)
        band_bins = band_bins[freqs[band_bins] <= max_freq_hz]
        if band_bins.size:
            selected_bins.extend(band_bins.tolist())
            peak_bin = int(band_bins[np.argmax(np.abs(fft_values[band_bins]))])
            retained_energy = float(np.sum(power[band_bins]))
            band_min = max(0.0, target - harmonic_bandwidth_hz)
            band_max = min(max_freq_hz, target + harmonic_bandwidth_hz)
            rows.append(
                {
                    "harmonic_order": order,
                    "target_frequency_Hz": float(target),
                    "selected_frequency_range_Hz": f"{band_min:.3f}-{band_max:.3f}",
                    "peak_frequency_in_band_Hz": float(freqs[peak_bin]),
                    "magnitude": float(np.abs(fft_values[peak_bin])),
                    "phase_radians": float(np.angle(fft_values[peak_bin])),
                    "retained_energy": retained_energy,
                    "retained_energy_ratio": retained_energy / total_energy,
                }
            )
        order += 1

    return np.unique(selected_bins).astype(int), pd.DataFrame(rows)


def build_energy_ratio_bins(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    min_freq_hz: float,
    max_freq_hz: float,
    energy_ratio: float,
) -> np.ndarray:
    power = np.abs(fft_values) ** 2
    candidate_bins = np.flatnonzero(frequency_mask(freqs, min_freq_hz, max_freq_hz))
    if candidate_bins.size == 0:
        return np.array([], dtype=int)

    ranked = candidate_bins[np.argsort(power[candidate_bins])[::-1]]
    target_energy = float(np.sum(power[candidate_bins]) * energy_ratio)
    cumulative = np.cumsum(power[ranked])
    keep_count = int(np.searchsorted(cumulative, target_energy, side="left") + 1)
    return np.sort(ranked[: max(1, keep_count)])


def build_peak_band_bins(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    min_freq_hz: float,
    max_freq_hz: float,
    peak_bandwidth_hz: float,
    peak_band_count: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    peak_bins = find_peak_bins(fft_values, freqs, min_freq_hz, max_freq_hz, peak_band_count)
    selected_bins: list[int] = []
    rows = []
    power = np.abs(fft_values) ** 2
    for rank, peak_bin in enumerate(peak_bins, start=1):
        center = float(freqs[peak_bin])
        band_bins = bins_in_band(freqs, center, peak_bandwidth_hz)
        band_bins = band_bins[frequency_mask(freqs[band_bins], min_freq_hz, max_freq_hz)]
        selected_bins.extend(band_bins.tolist())
        rows.append(
            {
                "rank": rank,
                "center_frequency_Hz": center,
                "selected_frequency_range_Hz": f"{center - peak_bandwidth_hz:.3f}-{center + peak_bandwidth_hz:.3f}",
                "magnitude": float(np.abs(fft_values[peak_bin])),
                "phase_radians": float(np.angle(fft_values[peak_bin])),
                "retained_energy": float(np.sum(power[band_bins])),
            }
        )
    return np.unique(selected_bins).astype(int), pd.DataFrame(rows)


def build_top_n_bins(
    fft_values: np.ndarray,
    freqs: np.ndarray,
    min_freq_hz: float,
    max_freq_hz: float,
    top_n: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    peak_bins = find_peak_bins(fft_values, freqs, min_freq_hz, max_freq_hz, top_n)
    rows = [
        {
            "rank": rank,
            "frequency_Hz": float(freqs[bin_index]),
            "magnitude": float(np.abs(fft_values[bin_index])),
            "phase_radians": float(np.angle(fft_values[bin_index])),
            "fft_bin": int(bin_index),
        }
        for rank, bin_index in enumerate(peak_bins, start=1)
    ]
    return peak_bins.astype(int), pd.DataFrame(rows)


def evaluate_selected_bins(
    original: np.ndarray,
    fft_values: np.ndarray,
    selected_bins: np.ndarray,
    signal_length: int,
    mode_name: str,
) -> dict[str, Any]:
    selected_bins = np.unique(selected_bins).astype(int)
    selected_fft = selected_fft_from_bins(fft_values, selected_bins)
    reconstructed = np.fft.irfft(selected_fft, n=signal_length)
    anti_noise = -reconstructed
    residual = original + anti_noise
    power = np.abs(fft_values) ** 2
    retained_energy_ratio = float(np.sum(power[selected_bins]) / (np.sum(power) + EPS)) if selected_bins.size else 0.0
    return {
        "mode": mode_name,
        "selected_bins": selected_bins,
        "reconstructed": reconstructed,
        "anti_noise": anti_noise,
        "residual": residual,
        "metrics": {
            "reconstruction_mode": mode_name,
            **compute_metrics(original, residual),
            "retained_bins": int(selected_bins.size),
            "retained_energy_ratio": retained_energy_ratio,
        },
    }


def _savefig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_original_waveform(original: np.ndarray, sample_rate: int, output_dir: Path) -> Path:
    sample_count = min(len(original), max(1, int(0.2 * sample_rate)))
    t = np.arange(sample_count) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t, original[:sample_count], linewidth=1)
    ax.set_title("Original Noise Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    return _savefig(fig, output_dir / "original_waveform.png")


def plot_original_spectrum_with_f0(
    freqs: np.ndarray,
    fft_values: np.ndarray,
    f0_hz: float,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(fft_values), linewidth=1)
    ax.axvline(f0_hz, color="crimson", linestyle="--", label=f"f0 = {f0_hz:.2f} Hz")
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Original Spectrum with Detected Fundamental Frequency")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "spectrum_before.png")


def plot_harmonic_spectrum(
    freqs: np.ndarray,
    fft_values: np.ndarray,
    harmonic_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(fft_values), linewidth=1)
    for _, row in harmonic_df.iterrows():
        ax.axvline(float(row["target_frequency_Hz"]), color="darkorange", alpha=0.45, linewidth=1)
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Harmonic Spectrum: f0, 2f0, 3f0, ...")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    return _savefig(fig, output_dir / "harmonic_spectrum.png")


def plot_selected_harmonic_bands(
    freqs: np.ndarray,
    fft_values: np.ndarray,
    harmonic_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(fft_values), linewidth=1)
    for _, row in harmonic_df.iterrows():
        band_min, band_max = [float(part) for part in row["selected_frequency_range_Hz"].split("-")]
        ax.axvspan(band_min, band_max, color="seagreen", alpha=0.18)
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Selected Harmonic Frequency Bands")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    return _savefig(fig, output_dir / "selected_harmonic_bands.png")


def plot_original_vs_reconstructed_waveform(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> Path:
    sample_count = min(len(original), max(1, int(0.1 * sample_rate)))
    t = np.arange(sample_count) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, original[:sample_count], label="Original noise", linewidth=1)
    ax.plot(t, reconstructed[:sample_count], label="Reconstructed main noise", linewidth=1)
    ax.set_title("Original vs. Reconstructed Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "original_vs_reconstructed_waveform.png")


def plot_original_vs_reconstructed_spectrum(
    freqs: np.ndarray,
    original_fft: np.ndarray,
    reconstructed: np.ndarray,
    output_dir: Path,
) -> Path:
    reconstructed_fft = np.fft.rfft(reconstructed)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(original_fft), label="Original", linewidth=1)
    ax.plot(freqs, np.abs(reconstructed_fft), label="Reconstructed", linewidth=1)
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Original vs. Reconstructed Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "original_vs_reconstructed_spectrum.png")


def plot_residual_spectrum(
    freqs: np.ndarray,
    residual: np.ndarray,
    output_dir: Path,
) -> Path:
    residual_fft = np.fft.rfft(residual)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(residual_fft), linewidth=1)
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Residual Spectrum After Cancellation")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    return _savefig(fig, output_dir / "residual_spectrum.png")


def plot_mode_comparison(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    comparison = metrics_df[metrics_df["reconstruction_mode"] != "ideal reference"].copy()
    label_map = {
        MODE_HARMONIC: "Harmonic bands",
        MODE_ENERGY: "Energy ratio",
        MODE_PEAK_BAND: "Peak bands",
        MODE_TOP_N: "Top N points",
    }
    comparison["plot_label"] = comparison["reconstruction_mode"].map(
        lambda value: label_map.get(value, value.replace(MODE_TOP_N, "Top N points"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(comparison["plot_label"], comparison["NRR_dB"])
    axes[0].set_title("NRR dB by Reconstruction Mode")
    axes[0].set_xlabel("NRR (dB)")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(comparison["plot_label"], comparison["MSE"])
    axes[1].set_title("MSE by Reconstruction Mode")
    axes[1].set_xlabel("MSE")
    axes[1].grid(True, axis="x", alpha=0.25)
    return _savefig(fig, output_dir / "mode_comparison.png")


def plot_delay_metrics(delay_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(delay_df["delay_ms"], delay_df["MSE"], marker="o")
    ax.set_title("Delay vs. MSE")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    delay_mse_path = _savefig(fig, output_dir / "delay_vs_mse.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(delay_df["delay_ms"], delay_df["NRR_dB"], marker="o")
    ax.set_title("Delay vs. NRR dB")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("NRR (dB)")
    ax.grid(True, alpha=0.3)
    delay_nrr_path = _savefig(fig, output_dir / "delay_vs_nrr.png")
    return {"delay_vs_mse": delay_mse_path, "delay_vs_nrr": delay_nrr_path}


def plot_delay_signal_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    delayed_outputs: dict[float, dict[str, np.ndarray]],
    sample_rate: int,
    output_dir: Path,
) -> Path:
    sample_count = min(len(original), max(1, int(0.1 * sample_rate)))
    t = np.arange(sample_count) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, original[:sample_count], label="Original noise", linewidth=1)
    axes[0].plot(t, reconstructed[:sample_count], label="Reconstructed main noise", linewidth=1)
    axes[0].set_title("Original and Reconstructed Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for delay_ms in [0.0, 1.0, 5.0, 10.0]:
        if delay_ms in delayed_outputs:
            signal = delayed_outputs[delay_ms]["delayed_anti_noise"]
            axes[1].plot(t, signal[:sample_count], label=f"Anti-noise delay {delay_ms:g} ms", linewidth=1)
    axes[1].set_title("Delayed Anti-noise Signals")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    return _savefig(fig, output_dir / "delayed_signal_comparison.png")


def plot_burst_waveform_comparison(
    original: np.ndarray,
    reduced: np.ndarray,
    burst_mask: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> Path:
    t = np.arange(len(original)) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, original, label="Original", linewidth=0.8, alpha=0.75)
    ax.plot(t, reduced, label="Burst reduced", linewidth=0.8)
    if np.any(burst_mask):
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t, ymin, ymax, where=burst_mask, color="crimson", alpha=0.12, label="Detected burst")
    ax.set_title("Burst Noise Reduction Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "burst_waveform_comparison.png")


def plot_burst_envelope(
    envelope: np.ndarray,
    threshold: float,
    burst_mask: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> Path:
    t = np.arange(len(envelope)) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, envelope, label="Short-time RMS envelope", linewidth=1)
    ax.axhline(threshold, color="crimson", linestyle="--", label="Burst threshold")
    if np.any(burst_mask):
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t, ymin, ymax, where=burst_mask, color="crimson", alpha=0.12, label="Detected burst")
    ax.set_title("Burst Detection Envelope")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "burst_detection_envelope.png")


def plot_burst_spectrum_comparison(
    original: np.ndarray,
    reduced: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> Path:
    freqs = np.fft.rfftfreq(len(original), d=1.0 / sample_rate)
    original_fft = np.fft.rfft(original)
    reduced_fft = np.fft.rfft(reduced)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.abs(original_fft), label="Original", linewidth=1)
    ax.plot(freqs, np.abs(reduced_fft), label="Burst reduced", linewidth=1)
    ax.set_xlim(0, min(5000, freqs[-1] if freqs.size else 5000))
    ax.set_title("Burst Reduction Spectrum Comparison")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _savefig(fig, output_dir / "burst_spectrum_comparison.png")


def run_delay_simulation(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    delays_ms: list[float] | None = None,
) -> dict[str, Any]:
    delays_ms = delays_ms or DEFAULT_DELAYS_MS
    rows = []
    delayed_outputs: dict[float, dict[str, np.ndarray]] = {}
    audio_paths: dict[str, Path] = {}
    for delay_ms in delays_ms:
        delay_samples = int((delay_ms / 1000.0) * sample_rate)
        delayed_anti_noise = zero_padded_delay(-reconstructed, delay_samples)
        residual = original + delayed_anti_noise
        delay_label = str(delay_ms).replace(".", "p")
        audio_paths[f"delayed_anti_noise_{delay_label}ms"] = write_wav(
            output_dir / f"delayed_anti_noise_{delay_label}ms.wav", sample_rate, delayed_anti_noise
        )
        audio_paths[f"delayed_residual_{delay_label}ms"] = write_wav(
            output_dir / f"delayed_residual_{delay_label}ms.wav", sample_rate, residual
        )
        delayed_outputs[delay_ms] = {
            "delayed_anti_noise": delayed_anti_noise,
            "delayed_residual": residual,
        }
        rows.append(
            {
                "delay_ms": delay_ms,
                "delay_samples": delay_samples,
                **compute_metrics(original, residual),
            }
        )
    delay_df = pd.DataFrame(rows)
    csv_path = output_dir / "delay_metrics.csv"
    delay_df.to_csv(csv_path, index=False)
    plot_paths = plot_delay_metrics(delay_df, output_dir)
    plot_paths["delayed_signal_comparison"] = plot_delay_signal_comparison(
        original, reconstructed, delayed_outputs, sample_rate, output_dir
    )
    return {
        "delay_metrics": delay_df,
        "csv_path": csv_path,
        "plot_paths": plot_paths,
        "audio_paths": audio_paths,
        "delayed_outputs": delayed_outputs,
    }


def run_burst_reduction(
    original: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    window_ms: float = DEFAULT_BURST_WINDOW_MS,
    threshold_multiplier: float = DEFAULT_BURST_THRESHOLD_MULTIPLIER,
    padding_ms: float = DEFAULT_BURST_PADDING_MS,
    attenuation: float = DEFAULT_BURST_ATTENUATION,
) -> dict[str, Any]:
    """Detect short transient bursts by RMS envelope and attenuate those regions."""
    window_samples = samples_from_ms(sample_rate, window_ms)
    padding_samples = samples_from_ms(sample_rate, padding_ms)
    envelope = moving_rms(original, window_samples)
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= EPS:
        threshold = median * threshold_multiplier
    else:
        threshold = median + threshold_multiplier * robust_sigma
    threshold = max(threshold, median + EPS)

    raw_mask = envelope > threshold
    burst_mask = pad_boolean_mask(raw_mask, padding_samples)
    reduced = original.copy()
    reduced[burst_mask] *= attenuation
    removed = original - reduced
    segments_df = mask_to_segments(burst_mask, sample_rate)
    segments_df["threshold"] = threshold
    segments_df["attenuation"] = attenuation
    segments_df["window_ms"] = window_ms
    segments_df["padding_ms"] = padding_ms

    metrics_df = pd.DataFrame(
        [
            {
                "burst_count": int(len(segments_df)),
                "threshold": threshold,
                "window_ms": window_ms,
                "padding_ms": padding_ms,
                "attenuation": attenuation,
                "RMS_original": float(np.sqrt(np.mean(original**2))),
                "RMS_burst_reduced": float(np.sqrt(np.mean(reduced**2))),
                "MSE_original_vs_reduced": float(np.mean((original - reduced) ** 2)),
                "NRR_dB": 10.0
                * math.log10((float(np.sum(original**2)) + EPS) / (float(np.sum(reduced**2)) + EPS)),
            }
        ]
    )

    segments_path = output_dir / "burst_segments.csv"
    metrics_path = output_dir / "burst_metrics.csv"
    segments_df.to_csv(segments_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    audio_paths = {
        "burst_reduced_noise": write_wav(output_dir / "burst_reduced_noise.wav", sample_rate, reduced),
        "burst_removed_noise": write_wav(output_dir / "burst_removed_noise.wav", sample_rate, removed),
    }
    plot_paths = {
        "burst_detection_envelope": plot_burst_envelope(envelope, threshold, burst_mask, sample_rate, output_dir),
        "burst_waveform_comparison": plot_burst_waveform_comparison(
            original, reduced, burst_mask, sample_rate, output_dir
        ),
        "burst_spectrum_comparison": plot_burst_spectrum_comparison(original, reduced, sample_rate, output_dir),
    }
    return {
        "burst_segments": segments_df,
        "burst_metrics": metrics_df,
        "csv_paths": {
            "burst_segments": segments_path,
            "burst_metrics": metrics_path,
        },
        "audio_paths": audio_paths,
        "plot_paths": plot_paths,
        "burst_reduced": reduced,
        "burst_removed": removed,
        "burst_mask": burst_mask,
        "burst_envelope": envelope,
    }


def run_analysis(
    sample_rate: int,
    original: np.ndarray,
    output_dir: str | Path = OUTPUT_DIR,
    min_freq_hz: float = 20.0,
    max_freq_hz: float = 3000.0,
    fundamental_max_freq_hz: float = 1000.0,
    reconstruction_mode: str = MODE_HARMONIC,
    harmonic_bandwidth_hz: float = 5.0,
    max_harmonics: int | None = None,
    energy_ratio: float = 0.9,
    peak_bandwidth_hz: float | None = None,
    peak_band_count: int = 10,
    top_n_values: list[int] | None = None,
    main_n: int = 10,
    delays_ms: list[float] | None = None,
    enable_burst_reduction: bool = True,
    burst_window_ms: float = DEFAULT_BURST_WINDOW_MS,
    burst_threshold_multiplier: float = DEFAULT_BURST_THRESHOLD_MULTIPLIER,
    burst_padding_ms: float = DEFAULT_BURST_PADDING_MS,
    burst_attenuation: float = DEFAULT_BURST_ATTENUATION,
    include_ideal: bool = True,
) -> dict[str, Any]:
    """Run FFT cancellation with the selected physical reconstruction mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = normalize_audio_array(original)
    if original.size == 0:
        raise ValueError("The uploaded WAV file contains no audio samples.")
    if min_freq_hz >= max_freq_hz:
        raise ValueError("Minimum frequency must be lower than maximum frequency.")
    if reconstruction_mode not in RECONSTRUCTION_MODES:
        raise ValueError(f"Unknown reconstruction mode: {reconstruction_mode}")

    signal_length = original.size
    freqs = np.fft.rfftfreq(signal_length, d=1.0 / sample_rate)
    fft_values = np.fft.rfft(original)
    power = np.abs(fft_values) ** 2
    analysis_max_hz = min(max_freq_hz, sample_rate / 2)
    fundamental_search_max = min(fundamental_max_freq_hz, analysis_max_hz)
    top_n_values = sorted(set(top_n_values or DEFAULT_TOP_N_VALUES))
    peak_bandwidth_hz = peak_bandwidth_hz or harmonic_bandwidth_hz

    f0 = detect_fundamental_frequency(fft_values, freqs, min_freq_hz, fundamental_search_max)
    f0_hz = float(f0["detected_f0_Hz"])
    fundamental_df = pd.DataFrame(
        [
            {
                "detected_f0_Hz": f0_hz,
                "search_range_Hz": f"{min_freq_hz:.3f}-{fundamental_search_max:.3f}",
                "magnitude": f0["magnitude"],
                "phase_radians": f0["phase_radians"],
            }
        ]
    )
    fundamental_csv_path = output_dir / "fundamental_frequency.csv"
    fundamental_df.to_csv(fundamental_csv_path, index=False)

    harmonic_bins, harmonic_df = build_harmonic_components(
        fft_values, freqs, f0_hz, analysis_max_hz, harmonic_bandwidth_hz, max_harmonics
    )
    harmonic_csv_path = output_dir / "harmonic_components.csv"
    harmonic_df.to_csv(harmonic_csv_path, index=False)

    energy_bins = build_energy_ratio_bins(fft_values, freqs, min_freq_hz, analysis_max_hz, energy_ratio)
    peak_band_bins, peak_band_df = build_peak_band_bins(
        fft_values, freqs, min_freq_hz, analysis_max_hz, peak_bandwidth_hz, peak_band_count
    )
    top_n_bins_by_value: dict[int, np.ndarray] = {}
    dominant_dfs = []
    for top_n in top_n_values:
        top_bins, top_df = build_top_n_bins(fft_values, freqs, min_freq_hz, analysis_max_hz, top_n)
        top_n_bins_by_value[top_n] = top_bins
        top_df = top_df.copy()
        top_df["top_n"] = top_n
        dominant_dfs.append(top_df)
    dominant_df = pd.concat(dominant_dfs, ignore_index=True) if dominant_dfs else pd.DataFrame()
    dominant_csv_path = output_dir / "dominant_frequencies.csv"
    dominant_df.to_csv(dominant_csv_path, index=False)

    selected_bins_by_mode = {
        MODE_HARMONIC: harmonic_bins,
        MODE_ENERGY: energy_bins,
        MODE_PEAK_BAND: peak_band_bins,
        MODE_TOP_N: top_n_bins_by_value.get(main_n, next(iter(top_n_bins_by_value.values()))),
    }

    evaluations: dict[str, dict[str, Any]] = {}
    metrics_rows = []
    for mode_name in [MODE_HARMONIC, MODE_ENERGY, MODE_PEAK_BAND, MODE_TOP_N]:
        evaluation = evaluate_selected_bins(original, fft_values, selected_bins_by_mode[mode_name], signal_length, mode_name)
        evaluations[mode_name] = evaluation
        metrics_rows.append(evaluation["metrics"])

    for top_n in top_n_values:
        evaluation = evaluate_selected_bins(
            original,
            fft_values,
            top_n_bins_by_value[top_n],
            signal_length,
            f"{MODE_TOP_N} N={top_n}",
        )
        metrics_rows.append(evaluation["metrics"])

    if include_ideal:
        ideal_residual = original + (-original)
        metrics_rows.append(
            {
                "reconstruction_mode": "ideal reference",
                **compute_metrics(original, ideal_residual),
                "retained_bins": "all",
                "retained_energy_ratio": 1.0,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    selected = evaluations[reconstruction_mode]
    selected_slug = {
        MODE_HARMONIC: "harmonic",
        MODE_ENERGY: "energy",
        MODE_PEAK_BAND: "peak_band",
        MODE_TOP_N: "top_n",
    }[reconstruction_mode]

    audio_outputs: dict[str, Path] = {
        "original": write_wav(output_dir / "original_normalized.wav", sample_rate, original),
        "reconstructed_main_noise": write_wav(
            output_dir / f"reconstructed_{selected_slug}.wav", sample_rate, selected["reconstructed"]
        ),
        "anti_noise": write_wav(output_dir / f"anti_noise_{selected_slug}.wav", sample_rate, selected["anti_noise"]),
        "residual_noise": write_wav(output_dir / f"residual_{selected_slug}.wav", sample_rate, selected["residual"]),
    }
    if include_ideal:
        audio_outputs["ideal_anti_noise"] = write_wav(output_dir / "ideal_anti_noise.wav", sample_rate, -original)
        audio_outputs["ideal_residual"] = write_wav(output_dir / "ideal_residual.wav", sample_rate, np.zeros_like(original))

    plot_paths = {
        "original_waveform": plot_original_waveform(original, sample_rate, output_dir),
        "spectrum_before": plot_original_spectrum_with_f0(freqs, fft_values, f0_hz, output_dir),
        "harmonic_spectrum": plot_harmonic_spectrum(freqs, fft_values, harmonic_df, output_dir),
        "selected_harmonic_bands": plot_selected_harmonic_bands(freqs, fft_values, harmonic_df, output_dir),
        "original_vs_reconstructed_waveform": plot_original_vs_reconstructed_waveform(
            original, selected["reconstructed"], sample_rate, output_dir
        ),
        "original_vs_reconstructed_spectrum": plot_original_vs_reconstructed_spectrum(
            freqs, fft_values, selected["reconstructed"], output_dir
        ),
        "residual_spectrum": plot_residual_spectrum(freqs, selected["residual"], output_dir),
        "mode_comparison": plot_mode_comparison(metrics_df, output_dir),
    }

    delay_result = run_delay_simulation(original, selected["reconstructed"], sample_rate, output_dir, delays_ms)
    plot_paths.update(delay_result["plot_paths"])
    audio_outputs.update(delay_result["audio_paths"])

    burst_result: dict[str, Any] | None = None
    if enable_burst_reduction:
        burst_result = run_burst_reduction(
            original,
            sample_rate,
            output_dir,
            window_ms=burst_window_ms,
            threshold_multiplier=burst_threshold_multiplier,
            padding_ms=burst_padding_ms,
            attenuation=burst_attenuation,
        )
        plot_paths.update(burst_result["plot_paths"])
        audio_outputs.update(burst_result["audio_paths"])

    return {
        "sample_rate": sample_rate,
        "duration": signal_length / sample_rate,
        "original": original,
        "freqs": freqs,
        "fft_values": fft_values,
        "fundamental_frequency": fundamental_df,
        "harmonic_components": harmonic_df,
        "peak_band_components": peak_band_df,
        "dominant_frequencies": dominant_df,
        "metrics": metrics_df,
        "delay_metrics": delay_result["delay_metrics"],
        "burst_metrics": burst_result["burst_metrics"] if burst_result else pd.DataFrame(),
        "burst_segments": burst_result["burst_segments"] if burst_result else pd.DataFrame(),
        "reconstruction_mode": reconstruction_mode,
        "selected_signals": {
            "reconstructed": selected["reconstructed"],
            "anti_noise": selected["anti_noise"],
            "residual": selected["residual"],
        },
        "selected_bins": selected["selected_bins"],
        "selected_metrics": selected["metrics"],
        "mode_evaluations": evaluations,
        "audio_outputs": audio_outputs,
        "csv_paths": {
            "fundamental_frequency": fundamental_csv_path,
            "harmonic_components": harmonic_csv_path,
            "dominant_frequencies": dominant_csv_path,
            "metrics": metrics_csv_path,
            "delay_metrics": delay_result["csv_path"],
            **(burst_result["csv_paths"] if burst_result else {}),
        },
        "plot_paths": plot_paths,
        "output_dir": output_dir,
        "analysis_parameters": {
            "min_freq_hz": min_freq_hz,
            "max_freq_hz": analysis_max_hz,
            "fundamental_max_freq_hz": fundamental_search_max,
            "harmonic_bandwidth_hz": harmonic_bandwidth_hz,
            "max_harmonics": max_harmonics,
            "energy_ratio": energy_ratio,
            "peak_bandwidth_hz": peak_bandwidth_hz,
            "peak_band_count": peak_band_count,
            "main_n": main_n,
            "enable_burst_reduction": enable_burst_reduction,
            "burst_window_ms": burst_window_ms,
            "burst_threshold_multiplier": burst_threshold_multiplier,
            "burst_padding_ms": burst_padding_ms,
            "burst_attenuation": burst_attenuation,
        },
    }


def print_summary(result: dict[str, Any]) -> None:
    f0 = result["fundamental_frequency"].iloc[0]["detected_f0_Hz"]
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Duration: {result['duration']:.3f} s")
    print(f"Detected fundamental frequency: {f0:.3f} Hz")
    print(f"Selected reconstruction mode: {result['reconstruction_mode']}")
    print("\nMetrics:")
    print(result["metrics"].to_string(index=False))
    print(f"\nOutputs saved to: {result['output_dir'].resolve()}")


def main() -> int:
    try:
        sample_rate, original = load_audio_mono(INPUT_WAV)
        result = run_analysis(sample_rate, original, output_dir=OUTPUT_DIR)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
