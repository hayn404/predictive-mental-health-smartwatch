"""
test_acousticFeatures.py
========================
Pytest test suite for Seren's acoustic feature extraction pipeline.

Test coverage areas:
  1. Data classes & serialisation
  2. Pitch (F0) extraction – voiced/unvoiced discrimination
  3. Energy – RMS, loudness, dynamic range, silence fraction
  4. Speech rate – syllable rate, pause detection
  5. Composite scores – arousal and valence proxy
  6. Full extract() integration on synthetic WAV fixtures
  7. Edge cases – silence, very short clips, mono/stereo
  8. Output schema – all expected keys present in to_dict()

Run:
  python generate_test_audio.py   # create fixtures (once)
  pytest test_acousticFeatures.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import acousticFeatures as af

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SR = 16_000
AUDIO_DIR = Path("test_audio")


def _sine(freq: float, duration: float, amplitude: float = 0.1, sr: int = SR) -> np.ndarray:
    """Generate a simple sine wave."""
    t = np.arange(int(duration * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration * sr), dtype=np.float32)


def _write_tmp(tmp_path: Path, y: np.ndarray, name: str, sr: int = SR) -> Path:
    p = tmp_path / name
    sf.write(str(p), y, sr)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data class & serialisation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataClasses:

    def test_pitch_features_fields(self):
        pf = af.PitchFeatures(
            mean_hz=150.0, std_hz=20.0, min_hz=100.0,
            max_hz=200.0, range_hz=100.0, voiced_fraction=0.7,
        )
        assert pf.mean_hz == 150.0
        assert pf.range_hz == pytest.approx(100.0)

    def test_energy_features_fields(self):
        ef = af.EnergyFeatures(
            rms_mean=0.05, rms_std=0.01, rms_max=0.15,
            loudness_lufs=-20.0, silence_fraction=0.1, dynamic_range_db=30.0,
        )
        assert ef.loudness_lufs == pytest.approx(-20.0)

    def test_speech_rate_fields(self):
        sr = af.SpeechRateFeatures(
            syllable_rate_per_sec=4.2, pause_count=3,
            pause_mean_duration_sec=0.3, total_pause_ratio=0.2,
            speaking_duration_sec=4.0, total_duration_sec=5.0,
        )
        assert sr.pause_count == 3

    def test_to_dict_returns_flat_dict(self, tmp_path):
        """to_dict() must return a flat key-value dict (no nested dicts)."""
        y = _sine(130, 2.0, amplitude=0.12)
        path = _write_tmp(tmp_path, y, "flat_dict.wav")
        result = af.extract(path)
        d = result.to_dict()
        assert isinstance(d, dict)
        for k, v in d.items():
            assert not isinstance(v, dict), f"Nested dict found at key '{k}'"

    def test_to_dict_contains_expected_keys(self, tmp_path):
        """to_dict() must include all 20+ required acoustic feature keys."""
        required_keys = [
            "pitch_mean_hz", "pitch_std_hz", "pitch_min_hz", "pitch_max_hz",
            "pitch_range_hz", "pitch_voiced_fraction",
            "energy_rms_mean", "energy_rms_std", "energy_rms_max",
            "energy_loudness_lufs", "energy_silence_fraction", "energy_dynamic_range_db",
            "speech_rate_syllable_rate_per_sec", "speech_rate_pause_count",
            "speech_rate_pause_mean_duration_sec", "speech_rate_total_pause_ratio",
            "speech_rate_speaking_duration_sec", "speech_rate_total_duration_sec",
            "arousal_score", "valence_proxy", "duration_sec",
        ]
        y = _sine(130, 2.0, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "keys.wav")
        d = af.extract(path).to_dict()
        for key in required_keys:
            assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pitch extraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPitchExtraction:

    def test_voiced_fraction_nonzero_for_voice(self, tmp_path):
        """Synthetic harmonic signal should yield non-zero voiced frames."""
        # Sawtooth-like voiced signal
        t = np.arange(3 * SR) / SR
        y = sum(0.1 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 6))
        y = y.astype(np.float32)
        path = _write_tmp(tmp_path, y, "voiced.wav")
        result = af.extract(path)
        assert result.pitch.voiced_fraction > 0.1, "Expected voiced frames in harmonic signal"

    def test_voiced_fraction_low_for_silence(self, tmp_path):
        """Pure silence should yield nearly zero voiced fraction."""
        y = _silence(3.0)
        path = _write_tmp(tmp_path, y, "silent.wav")
        result = af.extract(path)
        assert result.pitch.voiced_fraction < 0.15

    def test_pitch_range_nonnegative(self, tmp_path):
        """range_hz must always be ≥ 0."""
        y = _sine(200, 2.0, amplitude=0.15)
        path = _write_tmp(tmp_path, y, "range.wav")
        result = af.extract(path)
        assert result.pitch.range_hz >= 0.0

    def test_pitch_std_higher_for_variable_signal(self, tmp_path):
        """Signal with pitch variation should have higher std than monotone."""
        sr = SR
        t = np.arange(3 * sr) / sr
        # Monotone at 130 Hz
        mono = (0.1 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
        # Variable: modulate between 100–200 Hz
        f_mod = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)
        phase = 2 * np.pi * np.cumsum(f_mod) / sr
        var_sig = (0.1 * np.sin(phase)).astype(np.float32)

        mono_path = _write_tmp(tmp_path, mono, "mono.wav")
        var_path  = _write_tmp(tmp_path, var_sig, "variable.wav")

        mono_result = af.extract(mono_path)
        var_result  = af.extract(var_path)

        assert var_result.pitch.std_hz >= mono_result.pitch.std_hz, (
            "Variable pitch signal should have higher F0 std than monotone"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Energy extraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyExtraction:

    def test_rms_higher_for_louder_signal(self, tmp_path):
        """Louder amplitude must yield higher rms_mean."""
        quiet = _write_tmp(tmp_path, _sine(130, 2.0, amplitude=0.02), "quiet.wav")
        loud  = _write_tmp(tmp_path, _sine(130, 2.0, amplitude=0.20), "loud.wav")
        q_res = af.extract(quiet)
        l_res = af.extract(loud)
        assert l_res.energy.rms_mean > q_res.energy.rms_mean

    def test_silence_fraction_near_one_for_silence(self, tmp_path):
        """Silence clip should have silence_fraction close to 1.0."""
        path = _write_tmp(tmp_path, _silence(3.0) + np.random.randn(3 * SR) * 1e-5, "sil.wav")
        result = af.extract(path)
        assert result.energy.silence_fraction > 0.90

    def test_silence_fraction_low_for_loud_signal(self, tmp_path):
        """Loud voiced signal should have low silence_fraction."""
        path = _write_tmp(tmp_path, _sine(130, 3.0, amplitude=0.3), "loud2.wav")
        result = af.extract(path)
        assert result.energy.silence_fraction < 0.30

    def test_dynamic_range_nonnegative(self, tmp_path):
        """Dynamic range in dB must be ≥ 0."""
        path = _write_tmp(tmp_path, _sine(130, 2.0, amplitude=0.1), "dr.wav")
        result = af.extract(path)
        assert result.energy.dynamic_range_db >= 0.0

    def test_loudness_lufs_reasonable(self, tmp_path):
        """Loudness should be a negative dBFS value for normal speech amplitudes."""
        path = _write_tmp(tmp_path, _sine(130, 2.0, amplitude=0.08), "lufs.wav")
        result = af.extract(path)
        assert result.energy.loudness_lufs < 0, "Loudness should be negative dBFS"
        assert result.energy.loudness_lufs > -80, "Loudness should not be below -80 dBFS"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Speech rate tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeechRate:

    def test_total_duration_matches(self, tmp_path):
        """total_duration_sec must match the actual audio duration."""
        y = _sine(130, 4.5, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "dur.wav")
        result = af.extract(path)
        assert result.speech_rate.total_duration_sec == pytest.approx(4.5, abs=0.1)

    def test_pause_count_zero_for_continuous_signal(self, tmp_path):
        """Continuous voiced signal (no silent frames) should yield 0 pauses."""
        path = _write_tmp(tmp_path, _sine(130, 3.0, amplitude=0.3), "no_pauses.wav")
        result = af.extract(path)
        assert result.speech_rate.pause_count == 0

    def test_pause_ratio_high_for_silence(self, tmp_path):
        """Silence clip should have total_pause_ratio close to 1.0."""
        y = _silence(4.0) + np.random.randn(4 * SR) * 1e-5
        path = _write_tmp(tmp_path, y.astype(np.float32), "all_silence.wav")
        result = af.extract(path)
        assert result.speech_rate.total_pause_ratio > 0.85

    def test_speaking_duration_lte_total(self, tmp_path):
        """speaking_duration_sec must always be ≤ total_duration_sec."""
        y = np.concatenate([_sine(130, 1.5, amplitude=0.1), _silence(1.0),
                             _sine(130, 1.5, amplitude=0.1)])
        path = _write_tmp(tmp_path, y, "split.wav")
        result = af.extract(path)
        assert result.speech_rate.speaking_duration_sec <= result.speech_rate.total_duration_sec


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite score tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScores:

    def test_arousal_in_range(self, tmp_path):
        """arousal_score must be in [0, 1]."""
        path = _write_tmp(tmp_path, _sine(130, 3.0, amplitude=0.1), "ar.wav")
        result = af.extract(path)
        assert 0.0 <= result.arousal_score <= 1.0

    def test_valence_in_range(self, tmp_path):
        """valence_proxy must be in [0, 1]."""
        path = _write_tmp(tmp_path, _sine(130, 3.0, amplitude=0.1), "val.wav")
        result = af.extract(path)
        assert 0.0 <= result.valence_proxy <= 1.0

    def test_arousal_higher_for_loud_fast(self, tmp_path):
        """
        High-amplitude, syllable-rich signal should produce higher arousal
        than low-amplitude signal.
        """
        # Quiet slow signal
        quiet = _sine(130, 3.0, amplitude=0.02)
        # Loud fast-modulated signal
        t = np.arange(3 * SR) / SR
        loud_fast = (0.3 * np.sin(2 * np.pi * 200 * t) *
                     (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 6 * t)))).astype(np.float32)

        q_path = _write_tmp(tmp_path, quiet, "quiet_slow.wav")
        l_path = _write_tmp(tmp_path, loud_fast, "loud_fast.wav")

        q_res = af.extract(q_path)
        l_res = af.extract(l_path)

        assert l_res.arousal_score >= q_res.arousal_score, (
            f"Expected louder/faster signal to have higher arousal "
            f"({l_res.arousal_score:.3f} vs {q_res.arousal_score:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full integration tests on synthetic WAV fixtures
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """
    These tests use the synthetic fixtures generated by generate_test_audio.py.
    They validate the end-to-end pipeline on realistic speech-like signals.
    """

    @pytest.fixture(autouse=True)
    def check_fixtures(self):
        if not AUDIO_DIR.exists():
            pytest.skip(
                "Test audio fixtures not found. Run: python generate_test_audio.py"
            )

    def test_calm_speech_returns_valid_result(self):
        path = AUDIO_DIR / "calm_speech.wav"
        result = af.extract(path)
        assert isinstance(result, af.AcousticFeatureSet)
        assert result.duration_sec > 0
        assert result.pitch.voiced_fraction > 0  # should detect some voice
        assert result.speech_rate.total_duration_sec > 0

    def test_stressed_vs_calm_arousal(self):
        """Stressed speech should have higher arousal score than calm."""
        calm    = af.extract(AUDIO_DIR / "calm_speech.wav")
        stressed = af.extract(AUDIO_DIR / "stressed_speech.wav")
        assert stressed.arousal_score >= calm.arousal_score, (
            f"Stressed arousal {stressed.arousal_score:.3f} "
            f"should exceed calm {calm.arousal_score:.3f}"
        )

    def test_stressed_higher_energy(self):
        """Stressed speech clip must have higher RMS than calm."""
        calm    = af.extract(AUDIO_DIR / "calm_speech.wav")
        stressed = af.extract(AUDIO_DIR / "stressed_speech.wav")
        assert stressed.energy.rms_mean > calm.energy.rms_mean

    def test_monotone_lower_pitch_std(self):
        """Monotone clip must have lower F0 std than calm (variable) speech."""
        calm    = af.extract(AUDIO_DIR / "calm_speech.wav")
        mono    = af.extract(AUDIO_DIR / "monotone.wav")
        assert mono.pitch.std_hz <= calm.pitch.std_hz, (
            f"Monotone std {mono.pitch.std_hz:.2f} should be ≤ "
            f"calm std {calm.pitch.std_hz:.2f}"
        )

    def test_silence_clip_low_voiced_fraction(self):
        """Silence clip should have very low voiced fraction."""
        result = af.extract(AUDIO_DIR / "silence.wav")
        assert result.pitch.voiced_fraction < 0.15

    def test_silence_clip_high_silence_fraction(self):
        """Silence clip should have high silence fraction."""
        result = af.extract(AUDIO_DIR / "silence.wav")
        assert result.energy.silence_fraction > 0.90

    def test_silence_clip_scores_not_nan(self):
        """Even a silence clip should return valid (non-NaN) scores."""
        result = af.extract(AUDIO_DIR / "silence.wav")
        assert not math.isnan(result.arousal_score)
        assert not math.isnan(result.valence_proxy)

    def test_to_dict_values_are_finite(self):
        """All numeric values in to_dict() must be finite (no NaN / Inf)."""
        for clip in ["calm_speech", "stressed_speech", "silence", "monotone"]:
            path = AUDIO_DIR / f"{clip}.wav"
            d = af.extract(path).to_dict()
            for k, v in d.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"Non-finite value at {k} in {clip}: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_short_clip(self, tmp_path):
        """Sub-second clips must not raise exceptions."""
        y = _sine(130, 0.4, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "short.wav")
        result = af.extract(path)  # must not raise
        assert result.duration_sec == pytest.approx(0.4, abs=0.05)

    def test_low_amplitude_signal(self, tmp_path):
        """Very low amplitude signal (near noise floor) should not crash."""
        y = _sine(130, 2.0, amplitude=0.001)
        path = _write_tmp(tmp_path, y, "low_amp.wav")
        result = af.extract(path)  # must not raise
        assert result is not None

    def test_different_sample_rates(self, tmp_path):
        """
        Audio recorded at 44.1 kHz should be resampled and processed correctly.
        """
        sr_in = 44_100
        t = np.arange(int(2.0 * sr_in)) / sr_in
        y = (0.1 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
        path = tmp_path / "hires.wav"
        sf.write(str(path), y, sr_in)
        result = af.extract(path)  # must not raise
        assert result.sample_rate == 16_000  # resampled

    def test_white_noise_does_not_crash(self, tmp_path):
        """White noise should return valid (non-crashing) results."""
        np.random.seed(0)
        y = (np.random.randn(3 * SR) * 0.05).astype(np.float32)
        path = _write_tmp(tmp_path, y, "noise.wav")
        result = af.extract(path)  # must not raise
        assert 0.0 <= result.arousal_score <= 1.0

    def test_energy_rms_nonnegative(self, tmp_path):
        """RMS values must always be ≥ 0."""
        y = _sine(130, 2.0, amplitude=0.08)
        path = _write_tmp(tmp_path, y, "rms_nn.wav")
        result = af.extract(path)
        assert result.energy.rms_mean >= 0.0
        assert result.energy.rms_std >= 0.0
        assert result.energy.rms_max >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Schema compatibility with Seren voiceAnalysis output
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaCompatibility:
    """
    Verify that acoustic features can be merged with a mock voiceAnalysis
    result dict (simulating the VADER + LLM output from voiceAnalysis.ts).
    """

    MOCK_VADER_RESULT = {
        "transcript": "I feel really anxious today",
        "sentiment": {"compound": -0.45, "pos": 0.1, "neg": 0.4, "neu": 0.5},
        "emotion": "anxiety",
        "pii_stripped": False,
    }

    def test_merged_dict_has_no_key_conflicts(self, tmp_path):
        """
        Acoustic feature keys must not clash with voiceAnalysis result keys.
        """
        y = _sine(130, 2.0, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "merge.wav")
        acoustic = af.extract(path).to_dict()

        conflicts = set(acoustic.keys()) & set(self.MOCK_VADER_RESULT.keys())
        assert len(conflicts) == 0, f"Key conflicts between acoustic and VADER result: {conflicts}"

    def test_merged_dict_is_json_serialisable(self, tmp_path):
        """Combined acoustic + VADER dict must be JSON-serialisable."""
        import json
        y = _sine(130, 2.0, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "json.wav")
        acoustic = af.extract(path).to_dict()
        merged = {**self.MOCK_VADER_RESULT, **acoustic}
        json_str = json.dumps(merged)  # must not raise
        assert len(json_str) > 0

    def test_arousal_and_vader_compound_both_present(self, tmp_path):
        """
        After merging, both arousal_score (acoustic) and
        sentiment.compound (VADER) should be accessible.
        """
        y = _sine(130, 2.0, amplitude=0.1)
        path = _write_tmp(tmp_path, y, "combined.wav")
        acoustic = af.extract(path).to_dict()
        merged = {**self.MOCK_VADER_RESULT, **acoustic}

        assert "arousal_score" in merged
        assert "sentiment" in merged
