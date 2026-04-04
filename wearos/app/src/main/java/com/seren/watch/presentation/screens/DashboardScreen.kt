package com.seren.watch.presentation.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.DirectionsWalk
import androidx.compose.material.icons.automirrored.filled.TrendingUp
import androidx.compose.material.icons.filled.Bedtime
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Lightbulb
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.SelfImprovement
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberScalingLazyListState
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.Icon
import androidx.wear.compose.material.PositionIndicator
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.compose.material.Vignette
import androidx.wear.compose.material.VignettePosition
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.components.MetricChip
import com.seren.watch.presentation.components.StressArc
import com.seren.watch.presentation.theme.SerenColors

/**
 * Main dashboard screen — the first thing users see on the watch.
 * Shows stress gauge and all wellness metrics with navigation to detail screens.
 */
@Composable
fun DashboardScreen(
    viewModel: WellnessViewModel,
    onHeartRateTap: () -> Unit,
    onSleepTap: () -> Unit,
    onBreathingTap: () -> Unit,
    onSunlightTap: () -> Unit,
    onLocationTap: () -> Unit,
    onAnxietyTap: () -> Unit,
    onRecommendationsTap: () -> Unit,
    onInsightsTap: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current

    // Request body sensor permission
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            viewModel.onPermissionGranted()
        } else {
            viewModel.useMockData()
        }
    }

    LaunchedEffect(Unit) {
        val hasPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.BODY_SENSORS
        ) == PackageManager.PERMISSION_GRANTED

        if (hasPermission) {
            viewModel.onPermissionGranted()
        } else {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.BODY_SENSORS,
                    Manifest.permission.ACTIVITY_RECOGNITION,
                )
            )
        }
    }

    val listState = rememberScalingLazyListState()

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
            // App title
            item {
                Text(
                    text = "Seren",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = SerenColors.sageGreen,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Stress gauge
            item {
                StressArc(
                    score = state.stressScore,
                    level = state.stressLevel,
                    size = 110,
                    modifier = Modifier.padding(vertical = 4.dp),
                )
            }

            // Heart rate
            item {
                MetricChip(
                    icon = Icons.Default.Favorite,
                    iconColor = SerenColors.error,
                    label = "Heart Rate",
                    value = if (state.heartRate > 0) "${state.heartRate} BPM" else "-- BPM",
                    subtitle = if (state.isMonitoring) "Live" else "Tap to start",
                    onClick = onHeartRateTap,
                )
            }

            // Anxiety
            item {
                MetricChip(
                    icon = Icons.AutoMirrored.Filled.ShowChart,
                    iconColor = SerenColors.softBlue,
                    label = "Anxiety Index",
                    value = state.anxietyLevel.label,
                    subtitle = if (state.anxietySustained) "Sustained" else "Trend normal",
                    onClick = onAnxietyTap,
                )
            }

            // Sleep
            item {
                MetricChip(
                    icon = Icons.Default.Bedtime,
                    iconColor = SerenColors.softBlue,
                    label = "Sleep",
                    value = if (state.sleepQualityScore > 0) "${state.sleepQualityScore}%" else "--",
                    subtitle = if (state.sleepDurationMin > 0) {
                        "${state.sleepDurationMin / 60}h ${state.sleepDurationMin % 60}m"
                    } else "Last night",
                    onClick = onSleepTap,
                )
            }

            // Sunlight exposure
            item {
                MetricChip(
                    icon = Icons.Default.WbSunny,
                    iconColor = SerenColors.warning,
                    label = "Sunlight",
                    value = "${state.sunlightMinutes}m",
                    subtitle = if (state.isVitaminDWindow) "Vitamin D window open!" else "Goal: ${state.sunlightGoalMinutes}m",
                    onClick = onSunlightTap,
                )
            }

            // Location diversity
            item {
                MetricChip(
                    icon = Icons.Default.Place,
                    iconColor = SerenColors.violet,
                    label = "Location Diversity",
                    value = "${state.locationDiversityScore}/100",
                    subtitle = if (state.isMonotonousRoutine) "Try somewhere new!" else "${state.uniquePlacesVisited} places today",
                    onClick = onLocationTap,
                )
            }

            // Steps
            item {
                MetricChip(
                    icon = Icons.AutoMirrored.Filled.DirectionsWalk,
                    iconColor = SerenColors.sageGreen,
                    label = "Steps",
                    value = "${state.steps}",
                    subtitle = "${state.calories} cal",
                )
            }

            // Recommendations button
            item {
                Spacer(modifier = Modifier.height(4.dp))
                Chip(
                    onClick = onRecommendationsTap,
                    colors = ChipDefaults.chipColors(
                        backgroundColor = SerenColors.sageGreen.copy(alpha = 0.2f),
                    ),
                    label = {
                        Text(
                            "Recommendations",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium,
                        )
                    },
                    secondaryLabel = {
                        if (state.recommendations.isNotEmpty()) {
                            Text(
                                "${state.recommendations.size} suggestions for you",
                                fontSize = 10.sp,
                                color = SerenColors.textMuted,
                            )
                        }
                    },
                    icon = {
                        Icon(
                            Icons.Default.Lightbulb,
                            contentDescription = "Recommendations",
                            tint = SerenColors.sageGreen,
                            modifier = Modifier.size(18.dp),
                        )
                    },
                    modifier = Modifier.fillMaxWidth(0.9f),
                )
            }

            // Breathing exercise button
            item {
                Chip(
                    onClick = onBreathingTap,
                    colors = ChipDefaults.chipColors(
                        backgroundColor = SerenColors.violet.copy(alpha = 0.2f),
                    ),
                    label = {
                        Text(
                            "Breathing Exercise",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium,
                        )
                    },
                    icon = {
                        Icon(
                            Icons.Default.SelfImprovement,
                            contentDescription = "Breathe",
                            tint = SerenColors.violet,
                            modifier = Modifier.size(18.dp),
                        )
                    },
                    modifier = Modifier.fillMaxWidth(0.9f),
                )
            }

            // Weekly insights button
            item {
                Chip(
                    onClick = onInsightsTap,
                    colors = ChipDefaults.chipColors(
                        backgroundColor = SerenColors.softBlue.copy(alpha = 0.2f),
                    ),
                    label = {
                        Text(
                            "Weekly Insights",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium,
                        )
                    },
                    icon = {
                        Icon(
                            Icons.AutoMirrored.Filled.TrendingUp,
                            contentDescription = "Insights",
                            tint = SerenColors.softBlue,
                            modifier = Modifier.size(18.dp),
                        )
                    },
                    modifier = Modifier.fillMaxWidth(0.9f),
                )
            }
        }
    }
}
