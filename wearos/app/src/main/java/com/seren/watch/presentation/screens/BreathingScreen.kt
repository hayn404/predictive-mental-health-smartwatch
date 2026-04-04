package com.seren.watch.presentation.screens

import android.os.VibrationEffect
import android.os.Vibrator
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.ButtonDefaults
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import com.seren.watch.data.BREATHING_EXERCISES
import com.seren.watch.data.BreathingExercise
import com.seren.watch.presentation.theme.SerenColors
import kotlinx.coroutines.delay

/**
 * Guided breathing exercise with animated circle and haptic feedback.
 * Uses the 4-7-8 pattern by default.
 */
@Composable
fun BreathingScreen() {
    val exercise = BREATHING_EXERCISES[0] // 4-7-8 Calm
    var isActive by remember { mutableStateOf(false) }
    var currentPhase by remember { mutableStateOf("Ready") }
    var currentCycle by remember { mutableIntStateOf(0) }
    var secondsLeft by remember { mutableIntStateOf(0) }
    var breathScale by remember { mutableStateOf(0.4f) }

    val context = LocalContext.current
    val vibrator = remember { context.getSystemService(Vibrator::class.java) }

    val animatedScale by animateFloatAsState(
        targetValue = breathScale,
        animationSpec = tween(durationMillis = 800),
        label = "breathScale",
    )

    // Breathing cycle logic
    LaunchedEffect(isActive) {
        if (!isActive) return@LaunchedEffect

        for (cycle in 1..exercise.totalCycles) {
            currentCycle = cycle

            // Inhale
            currentPhase = "Breathe In"
            breathScale = 1f
            vibrator?.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
            for (s in exercise.inhaleSeconds downTo 1) {
                secondsLeft = s
                delay(1000L)
            }

            // Hold
            if (exercise.holdSeconds > 0) {
                currentPhase = "Hold"
                for (s in exercise.holdSeconds downTo 1) {
                    secondsLeft = s
                    delay(1000L)
                }
            }

            // Exhale
            currentPhase = "Breathe Out"
            breathScale = 0.4f
            vibrator?.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
            for (s in exercise.exhaleSeconds downTo 1) {
                secondsLeft = s
                delay(1000L)
            }

            // Hold after exhale (for box breathing)
            if (exercise.holdAfterExhaleSeconds > 0) {
                currentPhase = "Hold"
                for (s in exercise.holdAfterExhaleSeconds downTo 1) {
                    secondsLeft = s
                    delay(1000L)
                }
            }
        }

        // Done
        currentPhase = "Complete"
        vibrator?.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE))
        isActive = false
        currentCycle = 0
        breathScale = 0.4f
    }

    Scaffold(
        timeText = { TimeText() },
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
        ) {
            if (!isActive && currentPhase != "Complete") {
                // Start screen
                Text(
                    text = exercise.name,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = SerenColors.violet,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${exercise.inhaleSeconds}-${exercise.holdSeconds}-${exercise.exhaleSeconds}",
                    fontSize = 12.sp,
                    color = SerenColors.textSecondary,
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(
                    onClick = { isActive = true },
                    colors = ButtonDefaults.buttonColors(
                        backgroundColor = SerenColors.violet,
                    ),
                ) {
                    Text("Start", fontWeight = FontWeight.Bold)
                }
            } else if (isActive) {
                // Active breathing animation
                Text(
                    text = currentPhase,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = when (currentPhase) {
                        "Breathe In" -> SerenColors.sageGreen
                        "Hold" -> SerenColors.softBlue
                        "Breathe Out" -> SerenColors.violet
                        else -> SerenColors.textPrimary
                    },
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Animated breathing circle
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.size(80.dp),
                ) {
                    val circleColor = when (currentPhase) {
                        "Breathe In" -> SerenColors.sageGreen
                        "Hold" -> SerenColors.softBlue
                        "Breathe Out" -> SerenColors.violet
                        else -> SerenColors.textMuted
                    }

                    Canvas(modifier = Modifier.size((80 * animatedScale).dp)) {
                        drawCircle(
                            color = circleColor.copy(alpha = 0.3f),
                        )
                        drawCircle(
                            color = circleColor,
                            style = Stroke(width = 3.dp.toPx()),
                        )
                    }

                    Text(
                        text = "$secondsLeft",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.textPrimary,
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Cycle $currentCycle/${exercise.totalCycles}",
                    fontSize = 10.sp,
                    color = SerenColors.textMuted,
                )
            } else {
                // Complete
                Text(
                    text = "Well done!",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = SerenColors.sageGreen,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Take a moment to\nnotice how you feel",
                    fontSize = 12.sp,
                    color = SerenColors.textSecondary,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
