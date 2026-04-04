package com.seren.watch.presentation

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.seren.watch.data.AnxietyLevel
import com.seren.watch.data.StressLevel
import com.seren.watch.data.WatchRecommendation
import com.seren.watch.data.WellnessState
import com.seren.watch.health.HealthServicesManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

/**
 * ViewModel for the Seren Wear OS app.
 * Manages health data collection and wellness state.
 */
class WellnessViewModel(application: Application) : AndroidViewModel(application) {

    private val healthManager = HealthServicesManager(application)

    private val _state = MutableStateFlow(WellnessState())
    val state: StateFlow<WellnessState> = _state.asStateFlow()

    private val _isPermissionGranted = MutableStateFlow(false)
    val isPermissionGranted: StateFlow<Boolean> = _isPermissionGranted.asStateFlow()

    fun onPermissionGranted() {
        _isPermissionGranted.value = true
        startMonitoring()
    }

    private fun startMonitoring() {
        // Start real-time HR measurement
        viewModelScope.launch {
            healthManager.heartRateFlow()
                .catch { e ->
                    // Sensor not available — use mock data
                    useMockData()
                }
                .collect { bpm ->
                    val stress = healthManager.estimateStress(bpm)
                    _state.value = _state.value.copy(
                        heartRate = bpm,
                        stressScore = stress,
                        stressLevel = StressLevel.fromScore(stress),
                        isMonitoring = true,
                        lastUpdated = System.currentTimeMillis(),
                    )
                }
        }

        // Start passive monitoring (steps, calories)
        viewModelScope.launch {
            healthManager.startPassiveMonitoring()
        }

        // Collect full state updates
        viewModelScope.launch {
            healthManager.wellnessState.collect { healthState ->
                _state.value = _state.value.copy(
                    steps = healthState.steps,
                    calories = healthState.calories,
                )
            }
        }
    }

    /**
     * Use mock data when running on emulator or when sensors are unavailable.
     */
    fun useMockData() {
        val hour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        val isVitaminD = hour in 10..14

        // Simulate sunlight accumulation through the day
        val sunlightMin = when {
            hour < 7 -> 0
            hour < 10 -> 5
            hour < 13 -> 18
            hour < 16 -> 24
            else -> 32
        }

        _state.value = WellnessState(
            heartRate = 72,
            hrv = 55f,
            stressScore = 28,
            stressLevel = StressLevel.LOW,
            anxietyIndex = 15,
            anxietyLevel = AnxietyLevel.MINIMAL,
            anxietySustained = false,
            sleepQualityScore = 82,
            sleepDurationMin = 450,
            steps = 4230,
            calories = 180,
            spo2 = 98,
            skinTemp = 33.2f,

            // Sunlight
            sunlightMinutes = sunlightMin,
            sunlightGoalMinutes = 30,
            sunlightGoalProgress = (sunlightMin / 30f).coerceIn(0f, 1f),
            isVitaminDWindow = isVitaminD,

            // Location
            locationDiversityScore = 35,
            uniquePlacesVisited = 2,
            isMonotonousRoutine = true,

            // Recommendations
            recommendations = listOf(
                WatchRecommendation(
                    id = "outdoor_sunlight",
                    title = "Get Some Sunlight",
                    description = "Step outside for 15 minutes to boost serotonin.",
                    category = "outdoor",
                    durationMin = 15,
                    triggerReason = "Only ${sunlightMin}m of sunlight today",
                ),
                WatchRecommendation(
                    id = "explore_new_place",
                    title = "Explore Somewhere New",
                    description = "Break your routine — visit a new place.",
                    category = "exploration",
                    durationMin = 30,
                    triggerReason = "Your routine has been limited to the same places",
                ),
                WatchRecommendation(
                    id = "breathing_coherent",
                    title = "Coherent Breathing",
                    description = "Breathe at 5 breaths/min to boost HRV.",
                    category = "breathing",
                    durationMin = 10,
                    triggerReason = "Moderate stress detected",
                ),
            ),

            // Weekly trends
            weeklyStress = listOf(32, 45, 28, 52, 38, 25, 30),
            weeklyHrv = listOf(55, 48, 60, 42, 53, 62, 58),
            weeklySleep = listOf(78, 82, 70, 85, 75, 90, 82),
            weeklySunlight = listOf(25, 15, 35, 10, 28, 40, 32),

            isMonitoring = true,
        )
        _isPermissionGranted.value = true
    }
}
