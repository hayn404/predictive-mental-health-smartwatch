package com.seren.watch.presentation.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.PositionIndicator
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.compose.material.Vignette
import androidx.wear.compose.material.VignettePosition
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.theme.SerenColors

/**
 * Weekly insights/trends screen.
 * Shows mini bar charts for stress, sleep, HRV, and sunlight.
 */
@Composable
fun InsightsScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()
    val listState = rememberScalingLazyListState()
    val days = listOf("M", "T", "W", "T", "F", "S", "S")

    Scaffold(
        timeText = { TimeText() },
        vignette = { Vignette(vignettePosition = VignettePosition.TopAndBottom) },
        positionIndicator = { PositionIndicator(scalingLazyListState = listState) },
    ) {
        ScalingLazyColumn(
            state = listState,
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxSize(),
        ) {
            item {
                Text(
                    text = "Weekly Trends",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = SerenColors.sageGreen,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Stress trend
            item {
                MiniBarChart(
                    label = "Stress",
                    data = state.weeklyStress,
                    days = days,
                    color = SerenColors.violet,
                    maxValue = 100,
                )
            }

            // Sleep trend
            item {
                MiniBarChart(
                    label = "Sleep Quality",
                    data = state.weeklySleep,
                    days = days,
                    color = SerenColors.softBlue,
                    maxValue = 100,
                )
            }

            // HRV trend
            item {
                MiniBarChart(
                    label = "HRV (RMSSD)",
                    data = state.weeklyHrv,
                    days = days,
                    color = SerenColors.sageGreen,
                    maxValue = 80,
                )
            }

            // Sunlight trend
            item {
                MiniBarChart(
                    label = "Sunlight (min)",
                    data = state.weeklySunlight,
                    days = days,
                    color = SerenColors.warning,
                    maxValue = 60,
                )
            }
        }
    }
}

@Composable
private fun MiniBarChart(
    label: String,
    data: List<Int>,
    days: List<String>,
    color: Color,
    maxValue: Int,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.9f)
            .padding(vertical = 6.dp),
    ) {
        Text(
            text = label,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            color = SerenColors.textSecondary,
        )

        Spacer(modifier = Modifier.height(4.dp))

        // Bar chart
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(36.dp),
        ) {
            if (data.isEmpty()) return@Canvas

            val barCount = data.size.coerceAtMost(7)
            val totalWidth = size.width
            val barWidth = (totalWidth / barCount) * 0.6f
            val gap = (totalWidth / barCount) * 0.4f

            for (i in 0 until barCount) {
                val value = data[i].coerceIn(0, maxValue)
                val barHeight = (value.toFloat() / maxValue) * size.height
                val x = i * (barWidth + gap) + gap / 2

                // Bar background
                drawRoundRect(
                    color = SerenColors.surfaceLight,
                    topLeft = Offset(x, 0f),
                    size = Size(barWidth, size.height),
                    cornerRadius = CornerRadius(4.dp.toPx()),
                )

                // Bar fill
                drawRoundRect(
                    color = color,
                    topLeft = Offset(x, size.height - barHeight),
                    size = Size(barWidth, barHeight),
                    cornerRadius = CornerRadius(4.dp.toPx()),
                )
            }
        }

        Spacer(modifier = Modifier.height(2.dp))

        // Day labels
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            for (i in days.indices.take(data.size.coerceAtMost(7))) {
                Text(
                    text = days[i],
                    fontSize = 8.sp,
                    color = SerenColors.textMuted,
                    modifier = Modifier.width(16.dp),
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
