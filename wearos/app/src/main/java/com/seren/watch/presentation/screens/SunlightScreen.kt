package com.seren.watch.presentation.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.material.Icon
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.compose.material.Vignette
import androidx.wear.compose.material.VignettePosition
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.theme.SerenColors

/**
 * Sunlight exposure tracking screen.
 * Shows daily outdoor minutes, goal progress ring, and vitamin D window status.
 */
@Composable
fun SunlightScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()

    val progressColor = when {
        state.sunlightGoalProgress >= 1f -> SerenColors.sageGreen
        state.sunlightGoalProgress >= 0.5f -> SerenColors.warning
        else -> SerenColors.error
    }

    Scaffold(
        timeText = { TimeText() },
        vignette = { Vignette(vignettePosition = VignettePosition.TopAndBottom) },
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
        ) {
            Icon(
                Icons.Default.WbSunny,
                contentDescription = "Sunlight",
                tint = SerenColors.warning,
                modifier = Modifier.size(24.dp),
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Progress ring with minutes
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(90.dp),
            ) {
                Canvas(modifier = Modifier.size(90.dp)) {
                    val strokeWidth = 8.dp.toPx()
                    val arcSize = Size(size.width - strokeWidth, size.height - strokeWidth)
                    val topLeft = Offset(strokeWidth / 2, strokeWidth / 2)

                    // Background ring
                    drawArc(
                        color = SerenColors.surfaceLight,
                        startAngle = -90f,
                        sweepAngle = 360f,
                        useCenter = false,
                        topLeft = topLeft,
                        size = arcSize,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )

                    // Progress ring
                    drawArc(
                        color = progressColor,
                        startAngle = -90f,
                        sweepAngle = 360f * state.sunlightGoalProgress,
                        useCenter = false,
                        topLeft = topLeft,
                        size = arcSize,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "${state.sunlightMinutes}",
                        fontSize = 28.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.textPrimary,
                    )
                    Text(
                        text = "min",
                        fontSize = 10.sp,
                        color = SerenColors.textSecondary,
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Goal info
            Text(
                text = "Goal: ${state.sunlightGoalMinutes} min",
                fontSize = 12.sp,
                color = SerenColors.textSecondary,
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Vitamin D window status
            if (state.isVitaminDWindow) {
                Text(
                    text = "Vitamin D window is open!",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium,
                    color = SerenColors.sageGreen,
                    textAlign = TextAlign.Center,
                )
                Text(
                    text = "Go outside for 15 min",
                    fontSize = 10.sp,
                    color = SerenColors.textMuted,
                )
            } else {
                Text(
                    text = "Best time: 10 AM - 3 PM",
                    fontSize = 11.sp,
                    color = SerenColors.textMuted,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
