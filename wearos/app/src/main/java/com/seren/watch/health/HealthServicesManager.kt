package com.seren.watch.health

import android.content.Context
import android.util.Log
import androidx.health.services.client.HealthServices
import androidx.health.services.client.MeasureCallback
import androidx.health.services.client.MeasureClient
import androidx.health.services.client.PassiveListenerCallback
import androidx.health.services.client.PassiveMonitoringClient
import androidx.health.services.client.data.Availability
import androidx.health.services.client.data.DataPointContainer
import androidx.health.services.client.data.DataType
import androidx.health.services.client.data.DeltaDataType
import androidx.health.services.client.data.PassiveListenerConfig
import com.seren.watch.data.StressLevel
import com.seren.watch.data.WellnessState
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.guava.await
import kotlinx.coroutines.runBlocking
import java.util.concurrent.Executors

/**
 * Manages Health Services API on the watch.
 * Provides real-time heart rate, and passive monitoring for steps, calories, etc.
 */
class HealthServicesManager(private val context: Context) {

    companion object {
        private const val TAG = "SerenHealth"
    }

    private val healthClient = HealthServices.getClient(context)
    private val measureClient: MeasureClient = healthClient.measureClient
    private val passiveClient: PassiveMonitoringClient = healthClient.passiveMonitoringClient
    private val executor = Executors.newSingleThreadExecutor()

    private val _wellnessState = MutableStateFlow(WellnessState())
    val wellnessState: StateFlow<WellnessState> = _wellnessState.asStateFlow()

    /**
     * Check which data types are supported by this watch.
     */
    suspend fun getSupportedDataTypes(): Set<DataType<*, *>> {
        return try {
            val capabilities = measureClient.getCapabilitiesAsync().await()
            capabilities.supportedDataTypesMeasure
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get capabilities", e)
            emptySet()
        }
    }

    /**
     * Start real-time heart rate measurement.
     * Returns a Flow that emits HR values.
     */
    fun heartRateFlow(): Flow<Int> = callbackFlow {
        val callback = object : MeasureCallback {
            override fun onAvailabilityChanged(dataType: DeltaDataType<*, *>, availability: Availability) {
                Log.d(TAG, "HR availability: $availability")
            }

            override fun onDataReceived(data: DataPointContainer) {
                val hrPoints = data.getData(DataType.HEART_RATE_BPM)
                for (point in hrPoints) {
                    val bpm = point.value.toInt()
                    trySend(bpm)
                    updateState { copy(heartRate = bpm, lastUpdated = System.currentTimeMillis()) }
                }
            }
        }

        measureClient.registerMeasureCallback(DataType.HEART_RATE_BPM, executor, callback)
        Log.d(TAG, "HR measurement started")

        awaitClose {
            runBlocking {
                measureClient.unregisterMeasureCallbackAsync(DataType.HEART_RATE_BPM, callback)
            }
            Log.d(TAG, "HR measurement stopped")
        }
    }

    /**
     * Start passive monitoring for steps, calories, and daily metrics.
     */
    suspend fun startPassiveMonitoring() {
        try {
            val config = PassiveListenerConfig.builder()
                .setDataTypes(
                    setOf(
                        DataType.STEPS_DAILY,
                        DataType.CALORIES_DAILY,
                    )
                )
                .build()

            val callback = object : PassiveListenerCallback {
                override fun onNewDataPointsReceived(dataPoints: DataPointContainer) {
                    // Steps
                    dataPoints.getData(DataType.STEPS_DAILY).lastOrNull()?.let { point ->
                        updateState { copy(steps = point.value.toInt()) }
                    }
                    // Calories
                    dataPoints.getData(DataType.CALORIES_DAILY).lastOrNull()?.let { point ->
                        updateState { copy(calories = point.value.toInt()) }
                    }
                }
            }

            passiveClient.setPassiveListenerCallback(config, executor, callback)
            updateState { copy(isMonitoring = true) }
            Log.d(TAG, "Passive monitoring started")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start passive monitoring", e)
        }
    }

    /**
     * Compute a simple stress estimate from HR data.
     * Uses HR deviation from resting baseline as a proxy.
     * In production, this would use the XGBoost model with full HRV features.
     */
    fun estimateStress(currentHr: Int, restingHr: Int = 65): Int {
        if (currentHr <= 0) return 0
        val deviation = (currentHr - restingHr).coerceAtLeast(0)
        // Map deviation 0-60 to stress 0-100
        val score = ((deviation.toFloat() / 60f) * 100f).toInt().coerceIn(0, 100)
        updateState {
            copy(
                stressScore = score,
                stressLevel = StressLevel.fromScore(score),
            )
        }
        return score
    }

    private fun updateState(update: WellnessState.() -> WellnessState) {
        _wellnessState.value = _wellnessState.value.update()
    }
}
