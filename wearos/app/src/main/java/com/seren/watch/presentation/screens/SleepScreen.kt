package com.seren.watch.presentation.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import android.content.Intent
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bedtime
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.Icon
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.compose.material.Vignette
import androidx.wear.compose.material.VignettePosition
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.theme.SerenColors
import com.seren.watch.sleep.SleepCaptureService

/**
 * Sleep summary screen showing last night's sleep data.
 */
@Composable
fun SleepScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current
    var capturing by remember { mutableStateOf(false) }

    val hours = state.sleepDurationMin / 60
    val mins = state.sleepDurationMin % 60
    val qualityColor = when {
        state.sleepQualityScore >= 80 -> SerenColors.sageGreen
        state.sleepQualityScore >= 60 -> SerenColors.softBlue
        state.sleepQualityScore >= 40 -> SerenColors.warning
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
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
        ) {
            Icon(
                Icons.Default.Bedtime,
                contentDescription = "Sleep",
                tint = SerenColors.softBlue,
                modifier = Modifier.size(24.dp),
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Sleep quality
            Text(
                text = if (state.sleepQualityScore > 0) "${state.sleepQualityScore}%" else "--",
                fontSize = 42.sp,
                fontWeight = FontWeight.Bold,
                color = qualityColor,
            )
            Text(
                text = "Sleep Quality",
                fontSize = 12.sp,
                color = SerenColors.textSecondary,
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Duration
            if (state.sleepDurationMin > 0) {
                Text(
                    text = "${hours}h ${mins}m",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Medium,
                    color = SerenColors.textPrimary,
                )
                Text(
                    text = "Total Sleep",
                    fontSize = 10.sp,
                    color = SerenColors.textMuted,
                )
            } else {
                Text(
                    text = "No sleep data yet",
                    fontSize = 12.sp,
                    color = SerenColors.textMuted,
                    textAlign = TextAlign.Center,
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Start/stop the overnight HR + motion capture that streams to the phone.
            Chip(
                onClick = {
                    val intent = Intent(context, SleepCaptureService::class.java)
                    if (capturing) {
                        intent.action = SleepCaptureService.ACTION_STOP
                        context.startService(intent)
                    } else {
                        intent.action = SleepCaptureService.ACTION_START
                        ContextCompat.startForegroundService(context, intent)
                    }
                    capturing = !capturing
                },
                colors = ChipDefaults.chipColors(
                    backgroundColor = (if (capturing) SerenColors.error else SerenColors.sageGreen)
                        .copy(alpha = 0.2f),
                ),
                label = {
                    Text(
                        text = if (capturing) "Stop sleep tracking" else "Start sleep tracking",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                    )
                },
                icon = {
                    Icon(
                        imageVector = if (capturing) Icons.Default.Stop else Icons.Default.PlayArrow,
                        contentDescription = if (capturing) "Stop" else "Start",
                        tint = if (capturing) SerenColors.error else SerenColors.sageGreen,
                        modifier = Modifier.size(18.dp),
                    )
                },
                modifier = Modifier.fillMaxWidth(0.95f),
            )
        }
    }
}
