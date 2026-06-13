"""
Seren ML Pipeline — Step 2: Feature Engineering
=================================================
Extracts 29 HRV + biometric features per window from raw signals.
These exact same features must be replicated in TypeScript on the device.

Feature groups:
  - Time-domain HRV (9): meanRR, sdnn, rmssd, pnn50, pnn20, hrMean, hrStd, hrRange, cvRR
  - Frequency-domain HRV (7): vlfPower, lfPower, hfPower, lfHfRatio, totalPower, lfNorm, hfNorm
  - Non-linear HRV (5): sd1, sd2, sd1sd2Ratio, sampleEntropy, dfaAlpha1
  - Skin temperature (4): tempMean, tempSlope, tempStd, tempRange
  - Accelerometer (4): accelMagnitudeMean, accelMagnitudeStd, stepCount, activityType
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# BVP -> RR Intervals (Peak Detection)
# ============================================================

def bvp_to_rr_intervals(bvp: np.ndarray, fs: int = 64) -> np.ndarray:
    """
    Extract RR intervals from BVP (Blood Volume Pulse) signal.

    1. Bandpass filter (0.5-8 Hz) to remove noise
    2. Find systolic peaks
    3. Compute inter-beat intervals
    4. Remove artifacts (physiologically impossible intervals)

    Args:
        bvp: Raw BVP signal array
        fs: Sampling frequency (64 Hz for E4)

    Returns:
        Array of RR intervals in milliseconds
    """
    # Bandpass filter: 0.5-8 Hz (covers normal HR range 30-240 BPM)
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = min(8.0 / nyquist, 0.99)
    b, a = scipy_signal.butter(3, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, bvp)

    # Find peaks (systolic peaks in BVP)
    # min distance between peaks: ~0.3s (200 BPM max)
    min_distance = int(0.3 * fs)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance,
                                         height=np.percentile(filtered, 60))

    if len(peaks) < 2:
        return np.array([])

    # RR intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs * 1000

    # Artifact removal: keep only physiologically valid intervals
    # 300ms (200 BPM) to 2000ms (30 BPM)
    mask = (rr_intervals >= 300) & (rr_intervals <= 2000)

    # Also remove intervals that deviate >30% from rolling median
    if len(rr_intervals) > 5:
        rolling_median = np.median(rr_intervals)
        deviation_mask = np.abs(rr_intervals - rolling_median) / rolling_median < 0.30
        mask = mask & deviation_mask

    return rr_intervals[mask]


# ============================================================
# Time-Domain HRV Features (9 features)
# ============================================================

def compute_time_domain(rr: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain HRV features from RR intervals.

    Args:
        rr: RR intervals in milliseconds

    Returns:
        Dict with 9 time-domain features
    """
    if len(rr) < 5:
        return _empty_time_domain()

    nn_diffs = np.abs(np.diff(rr))
    hr = 60000 / rr  # Convert RR (ms) to HR (BPM)

    return {
        'meanRR': float(np.mean(rr)),
        'sdnn': float(np.std(rr, ddof=1)),
        'rmssd': float(np.sqrt(np.mean(nn_diffs ** 2))),
        'pnn50': float(np.sum(nn_diffs > 50) / len(nn_diffs) * 100),
        'pnn20': float(np.sum(nn_diffs > 20) / len(nn_diffs) * 100),
        'hrMean': float(np.mean(hr)),
        'hrStd': float(np.std(hr, ddof=1)),
        'hrRange': float(np.max(hr) - np.min(hr)),
        'cvRR': float(np.std(rr, ddof=1) / np.mean(rr)) if np.mean(rr) > 0 else 0,
    }


def _empty_time_domain() -> Dict[str, float]:
    return {k: 0.0 for k in [
        'meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20',
        'hrMean', 'hrStd', 'hrRange', 'cvRR'
    ]}


# ============================================================
# Frequency-Domain HRV Features (7 features)
# ============================================================

def compute_frequency_domain(rr: np.ndarray) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features using Welch's periodogram.

    Requires resampling RR intervals to uniform time grid first.

    Frequency bands:
      VLF: 0.003 - 0.04 Hz
      LF:  0.04 - 0.15 Hz
      HF:  0.15 - 0.4 Hz

    Args:
        rr: RR intervals in milliseconds

    Returns:
        Dict with 7 frequency-domain features
    """
    if len(rr) < 30:  # Need sufficient data for frequency analysis
        return _empty_frequency_domain()

    try:
        # Create cumulative time axis in seconds
        t = np.cumsum(rr) / 1000
        t = t - t[0]  # Start at 0

        # Resample to uniform 4 Hz grid via cubic interpolation
        fs_resample = 4  # Hz
        t_uniform = np.arange(t[0], t[-1], 1 / fs_resample)

        if len(t_uniform) < 30:
            return _empty_frequency_domain()

        interp_func = interp1d(t, rr, kind='cubic', fill_value='extrapolate')
        rr_resampled = interp_func(t_uniform)

        # Detrend
        rr_resampled = rr_resampled - np.mean(rr_resampled)

        # Welch's periodogram
        freqs, psd = scipy_signal.welch(
            rr_resampled,
            fs=fs_resample,
            nperseg=min(256, len(rr_resampled)),
            noverlap=min(128, len(rr_resampled) // 2),
        )

        # Frequency band power (integrate PSD)
        vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)

        _integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        vlf_power = float(_integrate(psd[vlf_mask], freqs[vlf_mask])) if vlf_mask.any() else 0
        lf_power = float(_integrate(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0
        hf_power = float(_integrate(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0

        total_power = vlf_power + lf_power + hf_power
        lf_hf_sum = lf_power + hf_power

        return {
            'vlfPower': vlf_power,
            'lfPower': lf_power,
            'hfPower': hf_power,
            'lfHfRatio': float(lf_power / hf_power) if hf_power > 0 else 0,
            'totalPower': total_power,
            'lfNorm': float(lf_power / lf_hf_sum * 100) if lf_hf_sum > 0 else 50,
            'hfNorm': float(hf_power / lf_hf_sum * 100) if lf_hf_sum > 0 else 50,
        }
    except Exception:
        return _empty_frequency_domain()


def _empty_frequency_domain() -> Dict[str, float]:
    return {k: 0.0 for k in [
        'vlfPower', 'lfPower', 'hfPower', 'lfHfRatio',
        'totalPower', 'lfNorm', 'hfNorm'
    ]}


# ============================================================
# Non-Linear HRV Features (5 features)
# ============================================================

def compute_nonlinear(rr: np.ndarray) -> Dict[str, float]:
    """
    Compute non-linear HRV features.

    - SD1/SD2: Poincare plot descriptors
    - Sample Entropy: Signal complexity
    - DFA alpha1: Detrended fluctuation analysis

    Args:
        rr: RR intervals in milliseconds

    Returns:
        Dict with 5 non-linear features
    """
    if len(rr) < 10:
        return _empty_nonlinear()

    try:
        # Poincare plot: SD1 and SD2
        rr1 = rr[:-1]
        rr2 = rr[1:]
        sd1 = float(np.std(rr2 - rr1, ddof=1) / np.sqrt(2))
        sd2 = float(np.std(rr2 + rr1, ddof=1) / np.sqrt(2))

        return {
            'sd1': sd1,
            'sd2': sd2,
            'sd1sd2Ratio': float(sd1 / sd2) if sd2 > 0 else 0,
            'sampleEntropy': _sample_entropy(rr, m=2, r_factor=0.2),
            'dfaAlpha1': _dfa_alpha1(rr),
        }
    except Exception:
        return _empty_nonlinear()


def _sample_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute sample entropy of RR interval series."""
    N = len(rr)
    if N < 20:
        return 0.0

    r = r_factor * np.std(rr, ddof=1)
    if r == 0:
        return 0.0

    def _count_matches(template_length):
        count = 0
        templates = np.array([rr[i:i + template_length] for i in range(N - template_length)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0:
        return 0.0

    return float(-np.log(A / B)) if A > 0 else 0.0


def _dfa_alpha1(rr: np.ndarray) -> float:
    """Compute DFA alpha1 (short-term scaling exponent, 4-16 beats)."""
    N = len(rr)
    if N < 16:
        return 0.0

    # Integrate the mean-centered series
    y = np.cumsum(rr - np.mean(rr))

    # Box sizes for alpha1: 4 to 16
    box_sizes = np.arange(4, min(17, N // 4 + 1))
    if len(box_sizes) < 2:
        return 0.0

    fluctuations = []
    for n in box_sizes:
        n_boxes = N // n
        if n_boxes == 0:
            continue

        rms_values = []
        for i in range(n_boxes):
            segment = y[i * n:(i + 1) * n]
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

        if rms_values:
            fluctuations.append(np.mean(rms_values))
        else:
            fluctuations.append(0)

    if len(fluctuations) < 2 or any(f <= 0 for f in fluctuations):
        return 0.0

    # Log-log fit
    log_n = np.log(box_sizes[:len(fluctuations)])
    log_f = np.log(np.array(fluctuations))

    valid = np.isfinite(log_n) & np.isfinite(log_f)
    if valid.sum() < 2:
        return 0.0

    coeffs = np.polyfit(log_n[valid], log_f[valid], 1)
    return float(coeffs[0])


def _empty_nonlinear() -> Dict[str, float]:
    return {k: 0.0 for k in ['sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1']}


# ============================================================
# Temperature Features (4 features)
# ============================================================

def compute_temperature_features(temp: np.ndarray, fs: int = 4) -> Dict[str, float]:
    """
    Compute skin temperature features from window.

    Args:
        temp: Temperature signal @ fs Hz
        fs: Sampling frequency

    Returns:
        Dict with 4 temperature features
    """
    if len(temp) < 2:
        return {'tempMean': 0, 'tempSlope': 0, 'tempStd': 0, 'tempRange': 0}

    t = np.arange(len(temp)) / fs / 60  # Time in minutes
    slope = np.polyfit(t, temp, 1)[0] if len(temp) > 1 else 0

    return {
        'tempMean': float(np.mean(temp)),
        'tempSlope': float(slope),         # °C per minute
        'tempStd': float(np.std(temp, ddof=1)) if len(temp) > 1 else 0,
        'tempRange': float(np.max(temp) - np.min(temp)),
    }


# ============================================================
# Accelerometer Features (4 features)
# ============================================================

def compute_accelerometer_features(acc: np.ndarray, fs: int = 32) -> Dict[str, float]:
    """
    Compute activity/movement features from 3-axis accelerometer.

    Args:
        acc: Accelerometer array (N, 3) @ fs Hz
        fs: Sampling frequency

    Returns:
        Dict with 4 accelerometer features
    """
    if len(acc) < 2:
        return {'accelMagnitudeMean': 0, 'accelMagnitudeStd': 0, 'stepCount': 0, 'activityType': 0}

    # Compute magnitude (Euclidean norm)
    magnitude = np.sqrt(np.sum(acc ** 2, axis=1))

    # Simple step estimation: count peaks above threshold in magnitude
    # Remove gravity (~1g ≈ 9.8 m/s²), but E4 outputs in g's
    # A step produces a spike in acceleration
    mag_detrended = magnitude - np.mean(magnitude)
    peaks, _ = scipy_signal.find_peaks(mag_detrended, height=0.3, distance=int(0.3 * fs))
    step_count = len(peaks)

    # Activity classification based on movement intensity
    mean_mag = float(np.mean(magnitude))
    if mean_mag < 1.02:  # Near 1g = gravity only = no movement
        activity = 0  # sedentary
    elif mean_mag < 1.1:
        activity = 1  # walking
    else:
        activity = 2  # active

    return {
        'accelMagnitudeMean': mean_mag,
        'accelMagnitudeStd': float(np.std(magnitude, ddof=1)),
        'stepCount': float(step_count),
        'activityType': float(activity),
    }


# ============================================================
# Master Feature Extraction
# ============================================================

# Feature names in exact order for model input
FEATURE_ORDER = [
    'meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20',
    'hrMean', 'hrStd', 'hrRange', 'cvRR',
    'vlfPower', 'lfPower', 'hfPower', 'lfHfRatio',
    'totalPower', 'lfNorm', 'hfNorm',
    'sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1',
    'tempMean', 'tempSlope', 'tempStd', 'tempRange',
    'accelMagnitudeMean', 'accelMagnitudeStd', 'stepCount', 'activityType',
]


def extract_features(window: Dict) -> Dict[str, float]:
    """
    Extract all 29 features from a single windowed segment.

    Args:
        window: Dict with 'bvp', 'eda', 'temp', 'acc' arrays

    Returns:
        Dict of 29 named features
    """
    # Step 1: BVP -> RR intervals
    rr = bvp_to_rr_intervals(window['bvp'], fs=64)

    # Step 2: Compute all feature groups
    features = {}
    features.update(compute_time_domain(rr))
    features.update(compute_frequency_domain(rr))
    features.update(compute_nonlinear(rr))
    features.update(compute_temperature_features(window['temp'], fs=4))
    features.update(compute_accelerometer_features(window['acc'], fs=32))

    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered numpy array for model input."""
    return np.array([features.get(name, 0.0) for name in FEATURE_ORDER])


def extract_features_dataframe(windows: List[Dict]) -> 'pd.DataFrame':
    """
    Extract features from all windows and return as DataFrame.

    Args:
        windows: List of window dicts from preprocessing

    Returns:
        DataFrame with feature columns + label columns
    """
    import pandas as pd

    rows = []
    for i, window in enumerate(windows):
        if i % 50 == 0:
            print(f'  Extracting features: {i}/{len(windows)}')

        features = extract_features(window)
        features['label'] = window.get('label', -1)
        features['label_name'] = window.get('label_name', 'unknown')
        features['stress_binary'] = window.get('stress_binary', -1)
        features['subject'] = window.get('subject', 'unknown')
        rows.append(features)

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    import pickle
    import pandas as pd
    from pathlib import Path

    # Load preprocessed windows
    windows_path = Path(__file__).parent.parent / 'data' / 'wesad_windows.pkl'
    if not windows_path.exists():
        print(f'Error: {windows_path} not found. Run preprocessing.py first.')
        exit(1)

    with open(windows_path, 'rb') as f:
        windows = pickle.load(f)

    print(f'Loaded {len(windows)} windows')
    print('Extracting features...')

    df = extract_features_dataframe(windows)

    # Save features
    output_path = Path(__file__).parent.parent / 'data' / 'wesad_features.csv'
    df.to_csv(output_path, index=False)
    print(f'Saved features to {output_path}')
    print(f'Shape: {df.shape}')
    print(f'\nLabel distribution:')
    print(df['label_name'].value_counts())
    print(f'\nStress binary distribution:')
    print(df['stress_binary'].value_counts())
