package com.seren.watch.env

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/** One ambient-light reading. */
data class LightSample(val timestampMs: Long, val lux: Float)

/** One GPS fix. */
data class LocationSample(
    val timestampMs: Long,
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float,
)

/**
 * On-watch environment sensors used for the location-diversity and sunlight-exposure
 * features. Exposes cold [Flow]s that register their underlying listener on collection
 * and unregister on cancellation — mirrors HealthServicesManager's sensor flows.
 */
class EnvironmentSensors(private val context: Context) {

    companion object {
        private const val TAG = "SerenEnv"
        /** GPS fix cadence — balanced power is plenty for home/work/novel-place clustering. */
        const val LOCATION_INTERVAL_MS = 5 * 60 * 1000L
        private const val LOCATION_MIN_INTERVAL_MS = 2 * 60 * 1000L
    }

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val fusedClient = LocationServices.getFusedLocationProviderClient(context)

    /** True when this watch exposes an ambient-light sensor (most do, for auto-brightness). */
    fun hasLightSensor(): Boolean =
        sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT) != null

    private fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    /** Emits ambient-light samples as the sensor reports changes. */
    fun lightFlow(): Flow<LightSample> = callbackFlow {
        val light = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        if (light == null) {
            Log.w(TAG, "No TYPE_LIGHT sensor on this device")
            close()
            return@callbackFlow
        }
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val deltaNanos = event.timestamp - System.nanoTime()
                val tsMs = System.currentTimeMillis() + deltaNanos / 1_000_000L
                trySend(LightSample(tsMs, event.values[0]))
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sensorManager.registerListener(listener, light, SensorManager.SENSOR_DELAY_NORMAL)
        Log.d(TAG, "Light capture started")
        awaitClose {
            sensorManager.unregisterListener(listener)
            Log.d(TAG, "Light capture stopped")
        }
    }

    /** Emits GPS fixes at [LOCATION_INTERVAL_MS] cadence (balanced power). */
    fun locationFlow(): Flow<LocationSample> = callbackFlow {
        if (!hasLocationPermission()) {
            Log.w(TAG, "Location permission not granted; no GPS")
            close()
            return@callbackFlow
        }
        val request = LocationRequest.Builder(
            Priority.PRIORITY_BALANCED_POWER_ACCURACY,
            LOCATION_INTERVAL_MS,
        ).setMinUpdateIntervalMillis(LOCATION_MIN_INTERVAL_MS).build()

        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                for (loc in result.locations) {
                    trySend(
                        LocationSample(
                            timestampMs = if (loc.time > 0) loc.time else System.currentTimeMillis(),
                            latitude = loc.latitude,
                            longitude = loc.longitude,
                            accuracy = if (loc.hasAccuracy()) loc.accuracy else -1f,
                        )
                    )
                }
            }
        }
        try {
            fusedClient.requestLocationUpdates(request, callback, Looper.getMainLooper())
            Log.d(TAG, "Location capture started @ ${LOCATION_INTERVAL_MS / 1000}s")
        } catch (e: SecurityException) {
            Log.e(TAG, "Location permission revoked at runtime", e)
            close(e)
            return@callbackFlow
        }
        awaitClose {
            fusedClient.removeLocationUpdates(callback)
            Log.d(TAG, "Location capture stopped")
        }
    }
}
