package com.seren.watch.env

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Foreground service that samples the watch's ambient-light sensor + GPS and ships
 * batches to the phone every [BATCH_INTERVAL_MS]. Light is throttled to ~1/min;
 * GPS arrives at the balanced-power cadence from [EnvironmentSensors].
 *
 * Start: startForegroundService(Intent(ctx, EnvironmentCaptureService::class.java))
 * Stop:  startService(Intent(ctx, EnvironmentCaptureService::class.java).setAction(ACTION_STOP))
 */
class EnvironmentCaptureService : Service() {

    companion object {
        private const val TAG = "SerenEnvSvc"
        const val ACTION_STOP = "com.seren.watch.env.STOP"
        const val CHANNEL_ID = "seren_env_capture"
        const val NOTIFICATION_ID = 2002

        /** How often to flush buffered samples to the phone. */
        private const val BATCH_INTERVAL_MS = 5 * 60 * 1000L
        /** Keep at most one light sample per this interval (sensor can fire in bursts). */
        private const val LIGHT_THROTTLE_MS = 60 * 1000L
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var captureJob: Job? = null
    private lateinit var sensors: EnvironmentSensors
    private lateinit var sender: WearableEnvSender

    private val lightBuf = ArrayDeque<LightSample>()
    private val locBuf = ArrayDeque<LocationSample>()
    private var lastLightKeptMs = 0L

    override fun onCreate() {
        super.onCreate()
        sensors = EnvironmentSensors(applicationContext)
        sender = WearableEnvSender(applicationContext)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopCapture()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(NOTIFICATION_ID, buildNotification())
        startCapture()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startCapture() {
        if (captureJob?.isActive == true) return
        lightBuf.clear(); locBuf.clear(); lastLightKeptMs = 0L

        captureJob = scope.launch {
            val lightJob = launch {
                sensors.lightFlow().collect { s ->
                    if (s.timestampMs - lastLightKeptMs >= LIGHT_THROTTLE_MS) {
                        lastLightKeptMs = s.timestampMs
                        synchronized(lightBuf) { lightBuf.addLast(s) }
                    }
                }
            }
            val locJob = launch {
                sensors.locationFlow().collect { s ->
                    synchronized(locBuf) { locBuf.addLast(s) }
                }
            }
            try {
                while (true) {
                    delay(BATCH_INTERVAL_MS)
                    flush()
                }
            } finally {
                lightJob.cancel()
                locJob.cancel()
            }
        }
        Log.d(TAG, "Environment capture started")
    }

    private suspend fun flush() {
        val light = synchronized(lightBuf) { val c = lightBuf.toList(); lightBuf.clear(); c }
        val loc = synchronized(locBuf) { val c = locBuf.toList(); locBuf.clear(); c }
        if (light.isNotEmpty()) sender.sendLight(light)
        if (loc.isNotEmpty()) sender.sendLocation(loc)
    }

    private fun stopCapture() {
        // Best-effort final flush of anything buffered.
        scope.launch { flush() }
        captureJob?.cancel()
        captureJob = null
        Log.d(TAG, "Environment capture stopped")
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Environment tracking",
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = "Captures ambient light + location for wellness insights" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Seren environment tracking")
            .setContentText("Sensing light & location")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .build()
}
