"""
acousticFeatures.py
===================
Seren — Acoustic Feature Extraction Pipeline
Extracts pitch (F0), energy (RMS/loudness), and speech rate
from a voice check-in audio clip (WAV / M4A / MP3).

Output dict is merged with the existing VADER / LLM sentiment
result inside voiceAnalysis.ts (via the Python sidecar or a
pre-processing step on the device before the Whisper call).

Author: Seren DSAI Team – Week 13
"""

from __future__ import annotations

import warnings
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as signal
import librosa

# ── silence irrelevant librosa warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PitchFeatures:
    """Fundamental frequency (F0) statistics over voiced frames."""
    mean_hz: float          # Mean F0 across voiced frames
    std_hz: float           # Std-dev of F0 (pitch variability / expressivity)
    min_hz: float           # Minimum F0 (lowest pitch)
    max_hz: float           # Maximum F0 (highest pitch)
    range_hz: float         # max − min  (dynamic range)
    voiced_fraction: float  # Fraction of frames classified as voiced (0–1)
    # Clinical note: monotone speech (low std) correlates with depression;
    # elevated mean + high variability may indicate anxiety / arousal.


@dataclass
class EnergyFeatures:
    """Short-term energy and loudness features."""
    rms_mean: float         # Mean RMS energy (linear, 0–1 normalised)
    rms_std: float          # Std-dev of RMS energy
    rms_max: float          # Peak RMS energy
    loudness_lufs: float    # Integrated loudness estimate (LUFS-like, EBU R128)
    silence_fraction: float # Fraction of frames below silence threshold
    dynamic_range_db: float # 95th pct − 5th pct of frame energy in dB
    # Clinical note: low RMS + high silence fraction correlates with low arousal /
    # fatigue; high RMS spike variability may indicate emotional dysregulation.


@dataclass
class SpeechRateFeatures:
    """Syllable-proxy and pause-based speech rate features."""
    syllable_rate_per_sec: float  # Estimated syllables per second
    pause_count: int              # Number of detected inter-word pauses
    pause_mean_duration_sec: float# Mean pause duration (seconds)
    total_pause_ratio: float      # Fraction of audio that is silence
    speaking_duration_sec: float  # Total voiced (non-pause) duration
    total_duration_sec: float     # Full clip duration
    # Clinical note: slow speech + long pauses = psychomotor retardation (depression);
    # fast speech + few pauses = mania / anxiety.


@dataclass
class AcousticFeatureSet:
    """Full acoustic feature bundle returned by extract()."""
    pitch: PitchFeatures
    energy: EnergyFeatures
    speech_rate: SpeechRateFeatures
    sample_rate: int
    duration_sec: float
    # Normalised composite scores (0–1, higher = more stressed/aroused)
    arousal_score: float    # Energy + speech-rate composite
    valence_proxy: float    # Pitch variability proxy (low = negative valence)

    def to_dict(self) -> dict:
        """Flat dict for JSON serialisation and ML feature merging."""
        d: dict = {}
        for field, val in asdict(self).items():
            if isinstance(val, dict):
                for k, v in val.items():
                    d[f"{field}_{k}"] = round(float(v), 6) if isinstance(v, float) else v
            else:
                d[field] = round(float(val), 6) if isinstance(val, float) else val
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Internal DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """
    Load any audio format supported by librosa (WAV, M4A, MP3, FLAC …).
    Resamples to `target_sr`, converts to mono.
    Returns (y, sr).
    """
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y, sr


def _extract_pitch(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> PitchFeatures:
    """
    Estimate F0 using librosa's pyin (probabilistic YIN) algorithm.
    pyin returns (f0, voiced_flag, voiced_probs).
    Unvoiced frames are excluded from statistics.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    voiced_f0 = f0[voiced_flag]
    voiced_fraction = float(voiced_flag.mean()) if len(voiced_flag) > 0 else 0.0

    if len(voiced_f0) == 0:
        # No voiced frames detected (silence / noise only)
        return PitchFeatures(
            mean_hz=0.0, std_hz=0.0, min_hz=0.0,
            max_hz=0.0, range_hz=0.0, voiced_fraction=voiced_fraction,
        )

    return PitchFeatures(
        mean_hz=float(np.mean(voiced_f0)),
        std_hz=float(np.std(voiced_f0)),
        min_hz=float(np.min(voiced_f0)),
        max_hz=float(np.max(voiced_f0)),
        range_hz=float(np.max(voiced_f0) - np.min(voiced_f0)),
        voiced_fraction=voiced_fraction,
    )


def _extract_energy(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    silence_threshold_db: float = -40.0,
) -> EnergyFeatures:
    """
    Compute short-term RMS energy and loudness proxy.
    silence_threshold_db: frames below this dBFS level are treated as silence.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Avoid log(0)
    rms_nonzero = np.where(rms > 1e-10, rms, 1e-10)
    rms_db = 20.0 * np.log10(rms_nonzero)

    silence_mask = rms_db < silence_threshold_db
    silence_fraction = float(silence_mask.mean())

    # EBU R128-style integrated loudness: mean of non-silent frames in dBFS
    nonsil_db = rms_db[~silence_mask]
    loudness_lufs = float(np.mean(nonsil_db)) if len(nonsil_db) > 0 else -70.0

    # Dynamic range: difference between loud and quiet parts
    p95 = float(np.percentile(rms_db, 95))
    p05 = float(np.percentile(rms_db, 5))
    dynamic_range_db = max(0.0, p95 - p05)

    return EnergyFeatures(
        rms_mean=float(np.mean(rms)),
        rms_std=float(np.std(rms)),
        rms_max=float(np.max(rms)),
        loudness_lufs=loudness_lufs,
        silence_fraction=silence_fraction,
        dynamic_range_db=dynamic_range_db,
    )


def _extract_speech_rate(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    silence_threshold_db: float = -40.0,
    min_pause_sec: float = 0.15,
) -> SpeechRateFeatures:
    """
    Estimate speech rate via onset-envelope syllable proxy and silence segmentation.

    Syllable proxy: count prominent peaks in the onset strength envelope.
    Pauses: contiguous silent frames longer than min_pause_sec.
    """
    total_duration = len(y) / sr

    # ── Syllable count proxy via onset peaks ─────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Normalise envelope
    if onset_env.max() > 0:
        onset_env_norm = onset_env / onset_env.max()
    else:
        onset_env_norm = onset_env

    # Detect prominent peaks (above 20% of max, with min distance ≈ 80 ms)
    min_dist_frames = max(1, int(0.08 * sr / hop_length))
    peaks, _ = signal.find_peaks(
        onset_env_norm,
        height=0.20,
        distance=min_dist_frames,
    )
    syllable_count = len(peaks)
    syllable_rate = syllable_count / total_duration if total_duration > 0 else 0.0

    # ── Pause segmentation from energy ──────────────────────────────────────
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_db = 20.0 * np.log10(np.where(rms > 1e-10, rms, 1e-10))
    is_silence = rms_db < silence_threshold_db

    min_pause_frames = max(1, int(min_pause_sec * sr / hop_length))

    # Find contiguous silence runs
    pauses: list[float] = []
    i = 0
    while i < len(is_silence):
        if is_silence[i]:
            j = i
            while j < len(is_silence) and is_silence[j]:
                j += 1
            run_len = j - i
            if run_len >= min_pause_frames:
                pause_dur = run_len * hop_length / sr
                pauses.append(pause_dur)
            i = j
        else:
            i += 1

    total_silence_sec = float(np.sum(pauses)) if pauses else 0.0
    speaking_duration = max(0.0, total_duration - total_silence_sec)

    return SpeechRateFeatures(
        syllable_rate_per_sec=round(syllable_rate, 3),
        pause_count=len(pauses),
        pause_mean_duration_sec=float(np.mean(pauses)) if pauses else 0.0,
        total_pause_ratio=total_silence_sec / total_duration if total_duration > 0 else 0.0,
        speaking_duration_sec=speaking_duration,
        total_duration_sec=total_duration,
    )


def _composite_scores(
    pitch: PitchFeatures,
    energy: EnergyFeatures,
    speech_rate: SpeechRateFeatures,
) -> tuple[float, float]:
    """
    Compute two normalised composite scores [0, 1]:

    arousal_score  — high energy + fast speech → high arousal / potential anxiety
    valence_proxy  — high pitch variability → positive/expressive; low → flat/depressive

    These are heuristic aggregates, NOT clinical diagnostics.
    """
    # ── Arousal: energy × speech-rate composite ──────────────────────────────
    # energy component: rms_mean normalised (typical voiced speech ≈ 0.01–0.3)
    e_norm = min(1.0, energy.rms_mean / 0.15)
    # speech-rate component: typical conversational rate ≈ 3–5 syl/s; max cap at 8
    sr_norm = min(1.0, speech_rate.syllable_rate_per_sec / 6.0)
    # silence fraction inverted (lots of silence = low arousal)
    activity = 1.0 - speech_rate.total_pause_ratio
    arousal = float(np.clip(0.4 * e_norm + 0.4 * sr_norm + 0.2 * activity, 0.0, 1.0))

    # ── Valence proxy: pitch std normalised ──────────────────────────────────
    # Expressive speech ≈ 30–80 Hz std; monotone ≈ 0–10 Hz std
    if pitch.voiced_fraction < 0.1:
        valence = 0.5  # insufficient voiced data — neutral
    else:
        valence = float(np.clip(pitch.std_hz / 60.0, 0.0, 1.0))

    return round(arousal, 4), round(valence, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract(
    audio_path: str | Path,
    target_sr: int = 16_000,
    frame_length: int = 2048,
    hop_length: int = 512,
    silence_threshold_db: float = -40.0,
    fmin: float = 50.0,
    fmax: float = 600.0,
    min_pause_sec: float = 0.15,
) -> AcousticFeatureSet:
    """
    Full acoustic feature extraction pipeline.

    Parameters
    ----------
    audio_path          : Path to audio file (WAV, M4A, MP3, FLAC …).
    target_sr           : Resample rate (16 kHz is optimal for speech).
    frame_length        : STFT frame size in samples (2048 ≈ 128 ms @ 16 kHz).
    hop_length          : Frame step size in samples (512 ≈ 32 ms @ 16 kHz).
    silence_threshold_db: dBFS below which a frame is considered silence.
    fmin / fmax         : F0 search range in Hz (50–600 covers all human voices).
    min_pause_sec       : Minimum silence duration to count as a pause.

    Returns
    -------
    AcousticFeatureSet  : Structured object with .to_dict() for JSON output.
    """
    y, sr = _load_audio(audio_path, target_sr)
    duration = len(y) / sr

    pitch   = _extract_pitch(y, sr, frame_length, hop_length, fmin, fmax)
    energy  = _extract_energy(y, sr, frame_length, hop_length, silence_threshold_db)
    speech  = _extract_speech_rate(y, sr, hop_length, silence_threshold_db, min_pause_sec)
    arousal, valence = _composite_scores(pitch, energy, speech)

    return AcousticFeatureSet(
        pitch=pitch,
        energy=energy,
        speech_rate=speech,
        sample_rate=sr,
        duration_sec=round(duration, 3),
        arousal_score=arousal,
        valence_proxy=valence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acousticFeatures.py <audio_file> [--json]")
        sys.exit(1)

    path = sys.argv[1]
    as_json = "--json" in sys.argv

    result = extract(path)

    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        p, e, s = result.pitch, result.energy, result.speech_rate
        print(f"\n{'='*60}")
        print(f"  Seren Acoustic Analysis: {Path(path).name}")
        print(f"{'='*60}")
        print(f"  Duration       : {result.duration_sec:.2f} s")
        print(f"\n  ── Pitch ──────────────────────────────────────────────")
        print(f"  Mean F0        : {p.mean_hz:.1f} Hz")
        print(f"  Std  F0        : {p.std_hz:.1f} Hz  (variability)")
        print(f"  Range          : {p.range_hz:.1f} Hz  ({p.min_hz:.1f}–{p.max_hz:.1f})")
        print(f"  Voiced frac    : {p.voiced_fraction:.1%}")
        print(f"\n  ── Energy ─────────────────────────────────────────────")
        print(f"  RMS mean       : {e.rms_mean:.4f}")
        print(f"  Loudness (LUFS): {e.loudness_lufs:.1f} dBFS")
        print(f"  Dynamic range  : {e.dynamic_range_db:.1f} dB")
        print(f"  Silence frac   : {e.silence_fraction:.1%}")
        print(f"\n  ── Speech Rate ────────────────────────────────────────")
        print(f"  Syllable rate  : {s.syllable_rate_per_sec:.2f} syl/s")
        print(f"  Pauses         : {s.pause_count}  (mean {s.pause_mean_duration_sec:.2f} s)")
        print(f"  Speaking time  : {s.speaking_duration_sec:.2f} s / {s.total_duration_sec:.2f} s")
        print(f"\n  ── Composite Scores ───────────────────────────────────")
        print(f"  Arousal score  : {result.arousal_score:.3f}  (0=calm, 1=high arousal)")
        print(f"  Valence proxy  : {result.valence_proxy:.3f}  (0=flat/low, 1=expressive)")
        print(f"{'='*60}\n")
