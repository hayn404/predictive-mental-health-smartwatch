"""
generate_test_audio.py
======================
Synthesise WAV fixtures for the acoustic-feature test suite.
Produces four controlled clips covering:
  1. calm_speech   – slow rate, low energy, moderate pitch
  2. stressed_speech – fast rate, high energy, elevated pitch
  3. silence        – pure silence (edge case)
  4. monotone       – constant pitch, low variability (depression-proxy)

Run once before pytest:  python generate_test_audio.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path

SR = 16_000          # 16 kHz
OUT = Path("test_audio")
OUT.mkdir(exist_ok=True)


def _voiced_segment(
    duration_sec: float,
    f0: float,
    amplitude: float,
    sr: int = SR,
    f0_jitter: float = 0.0,
) -> np.ndarray:
    """
    Generate a voiced segment with a sawtooth-like harmonic stack
    and optional pitch jitter to simulate natural speech variability.
    """
    t = np.arange(int(duration_sec * sr)) / sr
    y = np.zeros_like(t)
    # Sum first 8 harmonics (sawtooth approximation of voiced speech)
    for k in range(1, 9):
        # Add jitter to simulate natural pitch variation
        jitter = np.random.uniform(-f0_jitter, f0_jitter, len(t)) if f0_jitter > 0 else 0
        y += (amplitude / k) * np.sin(2 * np.pi * (f0 * k + jitter) * t)
    # Apply slight amplitude modulation to simulate syllable energy contour
    mod_freq = 4.0  # ~4 Hz = ~4 syllables/sec
    am = 0.6 + 0.4 * np.abs(np.sin(np.pi * mod_freq * t))
    return y * am


def _silence(duration_sec: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_sec * sr))


def _noise_burst(duration_sec: float, amplitude: float = 0.005, sr: int = SR) -> np.ndarray:
    return np.random.randn(int(duration_sec * sr)) * amplitude


def make_calm_speech() -> None:
    """
    Calm speech: ~3 syl/s, moderate energy (amplitude=0.08),
    F0 ≈ 130 Hz (typical adult male), moderate pitch variability.
    Duration: 5 s (3 voiced + 2 s of inter-word pauses)
    """
    segments = []
    np.random.seed(42)
    # 3 voiced bursts of ~0.9 s each, separated by 0.55 s pauses
    for _ in range(3):
        segments.append(_voiced_segment(0.90, f0=130, amplitude=0.08, f0_jitter=5.0))
        segments.append(_silence(0.55))
    # Trim to 5 s
    y = np.concatenate(segments)[:5 * SR]
    sf.write(OUT / "calm_speech.wav", y, SR)
    print("  ✓ calm_speech.wav")


def make_stressed_speech() -> None:
    """
    Stressed / anxious speech: ~6 syl/s, high energy (amplitude=0.25),
    elevated F0 ≈ 220 Hz (pitch rises under stress), high variability.
    Duration: 5 s (mostly voiced, short pauses)
    """
    segments = []
    np.random.seed(7)
    # 5 short voiced bursts of ~0.7 s each, 0.2 s micro-pauses
    for _ in range(5):
        segments.append(_voiced_segment(0.70, f0=220, amplitude=0.25, f0_jitter=20.0))
        segments.append(_silence(0.20))
    y = np.concatenate(segments)[:5 * SR]
    sf.write(OUT / "stressed_speech.wav", y, SR)
    print("  ✓ stressed_speech.wav")


def make_silence() -> None:
    """
    Pure silence + faint background noise (edge case — no voice).
    """
    y = _noise_burst(4.0, amplitude=0.0005)
    sf.write(OUT / "silence.wav", y, SR)
    print("  ✓ silence.wav")


def make_monotone() -> None:
    """
    Monotone speech: constant F0 (no jitter), low-moderate energy.
    Simulates flat affect associated with depression.
    """
    segments = []
    for _ in range(3):
        segments.append(_voiced_segment(1.0, f0=115, amplitude=0.06, f0_jitter=0.0))
        segments.append(_silence(0.5))
    y = np.concatenate(segments)[:5 * SR]
    sf.write(OUT / "monotone.wav", y, SR)
    print("  ✓ monotone.wav")


if __name__ == "__main__":
    print("Generating test audio fixtures …")
    make_calm_speech()
    make_stressed_speech()
    make_silence()
    make_monotone()
    print(f"Done — files written to ./{OUT}/")
