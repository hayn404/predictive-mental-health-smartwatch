package com.seren.watch.sleep

import com.seren.watch.health.AccelSample
import com.seren.watch.health.HrSample

/**
 * Aligned 30-second sleep epoch.
 *
 * `features` holds the 11 cache features in the order:
 *   hr_mean, hr_std, hr_min, hr_max, hr_range,
 *   hr_succdiff_std, hr_delta_prev,
 *   act_count, immobility_frac, act_max, act_std
 *
 * The 12th feature (time_of_night) is appended on the phone once the
 * full night is known. Do NOT add it here.
 */
data class RawEpochFeatures(
    /** Epoch start time, Unix ms. */
    val startMs: Long,
    /** Length 11. */
    val features: FloatArray,
) {
    companion object {
        const val SIZE = 11
    }
}

/**
 * Rolling 30-s epoch aggregator. Buffers raw HR and accel samples and emits
 * a [RawEpochFeatures] every time an epoch boundary is crossed.
 *
 * Mirrors the Python feature extraction in ml/sleep/prepare_features.py exactly:
 *   - HR features use a ±120-s window centred on the epoch midpoint
 *   - Accel features use only the in-epoch samples (start..start+30s)
 *   - acc_count / immob / act_max use abs-diff of magnitude; act_std uses raw magnitude std
 *
 * Thread-safety: not safe for concurrent calls. Drive from a single coroutine.
 */
class EpochAggregator(
    /** Absolute Unix ms at which capture began (used to align epoch boundaries). */
    captureStartMs: Long,
) {
    private val epochStartMs: Long = (captureStartMs / EPOCH_MS) * EPOCH_MS
    private var nextEpochIndex: Int = 0

    // Hold a generous trailing buffer so HR's ±120 s window is always satisfied.
    private val hrBuffer: ArrayDeque<HrSample> = ArrayDeque()
    private val accBuffer: ArrayDeque<AccelSample> = ArrayDeque()

    // Track previous epoch's hr_mean for hr_delta_prev.
    private var prevHrMean: Double? = null

    fun addHr(sample: HrSample) {
        hrBuffer.addLast(sample)
        // Trim: HR needs ±HR_WIN_MS of context, so only drop samples older than
        // (currentEpochCentre - HR_WIN_MS - EPOCH_MS) — be generous.
        val keepAfter = epochStartFor(nextEpochIndex) - HR_WIN_MS - EPOCH_MS
        while (hrBuffer.isNotEmpty() && hrBuffer.first().timestampMs < keepAfter) {
            hrBuffer.removeFirst()
        }
    }

    fun addAccel(sample: AccelSample) {
        accBuffer.addLast(sample)
        val keepAfter = epochStartFor(nextEpochIndex) - EPOCH_MS
        while (accBuffer.isNotEmpty() && accBuffer.first().timestampMs < keepAfter) {
            accBuffer.removeFirst()
        }
    }

    /**
     * Try to emit completed epochs. Returns 0..N epochs whose right edge has
     * passed (with a small safety margin for HR's forward window).
     */
    fun drainCompleted(nowMs: Long): List<RawEpochFeatures> {
        val emitted = mutableListOf<RawEpochFeatures>()
        while (true) {
            val et = epochStartFor(nextEpochIndex)
            // HR needs samples up to et + EPOCH_MS/2 + HR_WIN_MS.
            // We require that we've seen past that point before closing the epoch.
            val readyAt = et + EPOCH_MS / 2 + HR_WIN_MS
            if (nowMs < readyAt) break

            val features = extract(et)
            if (features != null) {
                emitted.add(RawEpochFeatures(et, features))
            }
            nextEpochIndex++
        }
        return emitted
    }

    private fun epochStartFor(index: Int): Long = epochStartMs + index.toLong() * EPOCH_MS

    /**
     * Returns the 11-feature vector for the epoch beginning at `et`, or null
     * if there is no HR data in the window (drop epoch, reset prevHrMean).
     */
    private fun extract(et: Long): FloatArray? {
        val centre = et + EPOCH_MS / 2
        // HR ±120 s
        val hrSeg = hrBuffer.asSequence()
            .filter { it.timestampMs in (centre - HR_WIN_MS) until (centre + HR_WIN_MS) }
            .map { it.bpm.toDouble() }
            .toList()
        if (hrSeg.isEmpty()) {
            prevHrMean = null
            return null
        }
        // Accel: in-epoch only
        val accSeg = accBuffer.asSequence()
            .filter { it.timestampMs in et until (et + EPOCH_MS) }
            .map { it.magnitudeG.toDouble() }
            .toList()

        val hrMean = hrSeg.average()
        val hrStd = if (hrSeg.size > 1) populationStd(hrSeg) else 0.0
        val hrMin = hrSeg.min()
        val hrMax = hrSeg.max()
        val hrRng = hrMax - hrMin
        val hrSdsd = if (hrSeg.size > 2) populationStd(diffs(hrSeg)) else 0.0
        val hrDlt = prevHrMean?.let { hrMean - it } ?: 0.0

        val (actCount, immob, actMax, actStd) = if (accSeg.size > 1) {
            val d = diffs(accSeg).map { kotlin.math.abs(it) }
            val count = d.sum()
            val frac = d.count { it < MOVE_THRESH_G } / d.size.toDouble()
            val max = d.max()
            val std = populationStd(accSeg)
            FourDoubles(count, frac, max, std)
        } else {
            FourDoubles(0.0, 0.0, 0.0, 0.0)
        }

        prevHrMean = hrMean
        return floatArrayOf(
            hrMean.toFloat(), hrStd.toFloat(), hrMin.toFloat(), hrMax.toFloat(), hrRng.toFloat(),
            hrSdsd.toFloat(), hrDlt.toFloat(),
            actCount.toFloat(), immob.toFloat(), actMax.toFloat(), actStd.toFloat(),
        )
    }

    private data class FourDoubles(val a: Double, val b: Double, val c: Double, val d: Double)

    companion object {
        const val EPOCH_MS: Long = 30_000L
        const val HR_WIN_MS: Long = 120_000L
        const val MOVE_THRESH_G: Double = 0.02
    }
}

// ── Numeric helpers ───────────────────────────────────────────────
// Population std (ddof=0) to match NumPy's default np.std.
private fun populationStd(xs: List<Double>): Double {
    val m = xs.average()
    var s = 0.0
    for (x in xs) s += (x - m) * (x - m)
    return kotlin.math.sqrt(s / xs.size)
}

private fun diffs(xs: List<Double>): List<Double> {
    val out = ArrayList<Double>(xs.size - 1)
    for (i in 1 until xs.size) out.add(xs[i] - xs[i - 1])
    return out
}
