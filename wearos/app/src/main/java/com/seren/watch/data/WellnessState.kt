package com.seren.watch.data

/**
 * Current wellness state displayed on the watch.
 * Updated by Health Services callbacks and optional phone sync.
 */
data class WellnessState(
    val heartRate: Int = 0,
    val hrv: Float = 0f,               // RMSSD in ms
    val stressScore: Int = 0,           // 0-100
    val stressLevel: StressLevel = StressLevel.LOW,
    val anxietyIndex: Int = 0,          // 0-100
    val anxietyLevel: AnxietyLevel = AnxietyLevel.MINIMAL,
    val anxietySustained: Boolean = false,
    val sleepQualityScore: Int = 0,     // 0-100
    val sleepDurationMin: Int = 0,
    val steps: Int = 0,
    val calories: Int = 0,
    val spo2: Int = 0,                  // 0-100%
    val skinTemp: Float = 0f,           // Celsius

    // Sunlight exposure
    val sunlightMinutes: Int = 0,       // Minutes outdoors today
    val sunlightGoalMinutes: Int = 30,
    val sunlightGoalProgress: Float = 0f, // 0-1
    val isVitaminDWindow: Boolean = false,

    // Location diversity
    val locationDiversityScore: Int = 0,  // 0-100
    val uniquePlacesVisited: Int = 0,
    val isMonotonousRoutine: Boolean = false,

    // Recommendations
    val recommendations: List<WatchRecommendation> = emptyList(),

    // Weekly trends (for insights)
    val weeklyStress: List<Int> = emptyList(),     // 7 values (Mon-Sun)
    val weeklyHrv: List<Int> = emptyList(),
    val weeklySleep: List<Int> = emptyList(),
    val weeklySunlight: List<Int> = emptyList(),

    val isMonitoring: Boolean = false,
    val lastUpdated: Long = System.currentTimeMillis(),
)

enum class StressLevel(val label: String) {
    LOW("Low"),
    MODERATE("Moderate"),
    ELEVATED("Elevated"),
    HIGH("High");

    companion object {
        fun fromScore(score: Int): StressLevel = when {
            score >= 70 -> HIGH
            score >= 50 -> ELEVATED
            score >= 30 -> MODERATE
            else -> LOW
        }
    }
}

enum class AnxietyLevel(val label: String) {
    MINIMAL("Minimal"),
    MILD("Mild"),
    MODERATE("Moderate"),
    SEVERE("Severe");

    companion object {
        fun fromIndex(index: Int): AnxietyLevel = when {
            index >= 70 -> SEVERE
            index >= 50 -> MODERATE
            index >= 30 -> MILD
            else -> MINIMAL
        }
    }
}

/**
 * A simplified recommendation for display on the watch.
 */
data class WatchRecommendation(
    val id: String,
    val title: String,
    val description: String,
    val category: String,       // breathing, physical, outdoor, exploration, etc.
    val durationMin: Int,
    val triggerReason: String,
)

/**
 * Breathing exercise configuration.
 */
data class BreathingExercise(
    val name: String,
    val inhaleSeconds: Int,
    val holdSeconds: Int,
    val exhaleSeconds: Int,
    val holdAfterExhaleSeconds: Int = 0,
    val totalCycles: Int = 4,
)

val BREATHING_EXERCISES = listOf(
    BreathingExercise("4-7-8 Calm", inhaleSeconds = 4, holdSeconds = 7, exhaleSeconds = 8, totalCycles = 4),
    BreathingExercise("Box Breathing", inhaleSeconds = 4, holdSeconds = 4, exhaleSeconds = 4, holdAfterExhaleSeconds = 4, totalCycles = 6),
    BreathingExercise("Coherent", inhaleSeconds = 6, holdSeconds = 0, exhaleSeconds = 6, totalCycles = 10),
)
