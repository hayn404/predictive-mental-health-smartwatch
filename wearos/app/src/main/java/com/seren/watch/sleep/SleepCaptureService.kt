package com.seren.watch.sleep

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.seren.watch.health.HealthServicesManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch

/**
 * Foreground service that runs while the user is sleeping. Pumps HR + accel
 * samples into an EpochAggregator, batches completed epochs, and ships them
 * to the phone every BATCH_INTERVAL_MS. On stop, flushes a final batch with
 * the "finalize" marker so the phone runs inference and clears state.
 *
 * Start:  startForegroundService(Intent(ctx, SleepCaptureService::class.java).setAction(ACTION_START))
 * Stop:   startService(Intent(ctx, SleepCaptureService::class.java).setAction(ACTION_STOP))
 */
class SleepCaptureService : Service() {

    companion object {
        private const val TAG = "SerenSleepSvc"
        const val ACTION_START = "com.seren.watch.sleep.START"
        const val ACTION_STOP = "com.seren.watch.sleep.STOP"
        const val CHANNEL_ID = "seren_sleep_capture"
        const val NOTIFICATION_ID = 2001

        /** How often to flush completed epochs to the phone. */
        private const val BATCH_INTERVAL_MS: Long = 10 * 60 * 1000L
        /** Sweep cadence for draining the aggregator (must be < BATCH_INTERVAL_MS). */
        private const val SWEEP_INTERVAL_MS: Long = 30_000L
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var captureJob: Job? = null
    private lateinit var health: HealthServicesManager
    private lateinit var sender: WearableFeatureSender
    private lateinit var aggregator: EpochAggregator
    private var captureStartMs: Long = 0L
    private val pending = ArrayDeque<RawEpochFeatures>()
    private var lastFlushMs: Long = 0L

    override fun onCreate() {
        super.onCreate()
        health = HealthServicesManager(applicationContext)
        sender = WearableFeatureSender(applicationContext)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopCapture()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                startForeground(NOTIFICATION_ID, buildNotification())
                startCapture()
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startCapture() {
        if (captureJob?.isActive == true) return
        captureStartMs = System.currentTimeMillis()
        lastFlushMs = captureStartMs
        aggregator = EpochAggregator(captureStartMs)
        pending.clear()

        captureJob = scope.launch {
            // HR pump
            val hrJob = launch {
                health.heartRateFlow().collect { bpm ->
                    aggregator.addHr(com.seren.watch.health.HrSample(System.currentTimeMillis(), bpm))
                }
            }
            // Accel pump
            val accJob = launch {
                health.accelerometerFlow().collect { sample ->
                    aggregator.addAccel(sample)
                }
            }
            // Sweep + batch loop. Cancellation propagates from stopCapture()
            // via the scope cancel; the child collectors above stop with it.
            try {
                while (true) {
                    delay(SWEEP_INTERVAL_MS)
                    val now = System.currentTimeMillis()
                    val completed = aggregator.drainCompleted(now)
                    if (completed.isNotEmpty()) pending.addAll(completed)
                    if (pending.isNotEmpty() && now - lastFlushMs >= BATCH_INTERVAL_MS) {
                        val batch = pending.toList()
                        pending.clear()
                        sender.send(captureStartMs, batch, final = false)
                        lastFlushMs = now
                    }
                }
            } finally {
                hrJob.cancel()
                accJob.cancel()
            }
        }
        Log.d(TAG, "Sleep capture started at $captureStartMs")
    }

    private fun stopCapture() {
        val now = System.currentTimeMillis()
        // Final flush: include anything still in the aggregator.
        val final = (pending + aggregator.drainCompleted(now + EpochAggregator.HR_WIN_MS)).toList()
        pending.clear()
        scope.launch { sender.send(captureStartMs, final, final = true) }
        captureJob?.cancel()
        captureJob = null
        Log.d(TAG, "Sleep capture stopped; final batch ${final.size} epochs queued")
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Sleep tracking",
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = "Captures HR + motion for nightly sleep staging" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Seren sleep tracking")
            .setContentText("Capturing your sleep")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()
    }
}
