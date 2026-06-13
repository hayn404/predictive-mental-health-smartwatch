package com.seren.watch.health

/** Raw heart-rate sample emitted by HealthServicesManager.heartRateFlow(). */
data class HrSample(
    /** Unix epoch milliseconds. */
    val timestampMs: Long,
    /** Beats per minute. */
    val bpm: Int,
)

/**
 * Raw accelerometer magnitude sample emitted by HealthServicesManager.accelerometerFlow().
 * `magnitudeG` is sqrt(ax² + ay² + az²) in g units, **including gravity** (~1g at rest).
 * This matches `acc_magnitude` in ml/sleep/prepare_features.py.
 */
data class AccelSample(
    val timestampMs: Long,
    val magnitudeG: Float,
)
