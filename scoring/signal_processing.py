import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft


def estimate_sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[dt > 0]

    if len(dt) == 0:
        raise ValueError("Cannot estimate sampling rate from timestamp_s.")

    return 1.0 / np.median(dt)


def lowpass_filter_signal(
    x: np.ndarray,
    fs: float,
    cutoff: float = 3.0,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth Low-pass Filter.
    노이즈성 고주파 성분은 줄이고, 전체 주행 흐름은 유지한다.
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 10:
        return x

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1.0:
        return x

    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, x)


def fft_high_frequency_ratio(
    x: np.ndarray,
    fs: float,
    high_freq_threshold: float = 0.5,
) -> float:
    """
    FFT High Frequency Ratio.
    전체 주행 구간에서 고주파 에너지가 얼마나 큰지 계산한다.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    fft_values = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)

    power = np.abs(fft_values) ** 2
    total_power = np.sum(power)

    if total_power == 0:
        return 0.0

    high_power = np.sum(power[freqs >= high_freq_threshold])
    return float(high_power / total_power)


def stft_event_ratio(
    x: np.ndarray,
    fs: float,
    high_freq_threshold: float = 0.5,
    hfr_percentile: float = 90,
    energy_percentile: float = 70,
) -> float:
    """
    STFT Event Ratio.
    시간 구간별로 고주파 비율과 에너지 크기를 함께 보고 이벤트 후보를 계산한다.

    Event Candidate 조건:
    1. High Frequency Ratio가 상위 90% 이상
    2. Total Energy가 상위 70% 이상
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    if len(x) < 20:
        return 0.0

    nperseg = min(128, len(x))
    noverlap = nperseg // 2

    freqs, times, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2

    total_energy = np.sum(power, axis=0)
    high_energy = np.sum(power[freqs >= high_freq_threshold, :], axis=0)

    hfr = np.divide(
        high_energy,
        total_energy,
        out=np.zeros_like(high_energy),
        where=total_energy != 0,
    )

    if len(hfr) == 0:
        return 0.0

    hfr_threshold = np.percentile(hfr, hfr_percentile)
    energy_threshold = np.percentile(total_energy, energy_percentile)

    events = (hfr >= hfr_threshold) & (total_energy >= energy_threshold)
    return float(np.mean(events))


def extract_signal_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    각 feature에 대해 Low-pass, FFT HFR, STFT Event Ratio를 계산한다.
    """
    time_s = df["timestamp_s"].to_numpy()
    fs = estimate_sampling_rate(time_s)

    rows = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        raw = df[col].to_numpy(dtype=float)
        filtered = lowpass_filter_signal(raw, fs)

        rows.append({
            "feature": col,
            "sampling_rate_hz": fs,
            "mean": float(np.mean(filtered)),
            "std": float(np.std(filtered)),
            "rms": float(np.sqrt(np.mean(filtered ** 2))),
            "fft_high_freq_ratio": fft_high_frequency_ratio(filtered, fs),
            "stft_event_ratio": stft_event_ratio(filtered, fs),
        })

    return pd.DataFrame(rows)