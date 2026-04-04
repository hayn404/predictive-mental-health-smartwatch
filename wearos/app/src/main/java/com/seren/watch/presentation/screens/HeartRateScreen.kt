package com.seren.watch.presentation.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.MonitorHeart
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
 * Detailed heart rate view with large BPM display and HRV.
 */
@Composable
fun HeartRateScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()

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
            // Heart icon
            Icon(
                Icons.Default.Favorite,
                contentDescription = "Heart",
                tint = SerenColors.error,
                modifier = Modifier.size(28.dp),
            )

            Spacer(modifier = Modifier.height(8.dp))

            // BPM value
            Text(
                text = if (state.heartRate > 0) "${state.heartRate}" else "--",
                fontSize = 48.sp,
                fontWeight = FontWeight.Bold,
                color = SerenColors.textPrimary,
            )
            Text(
                text = "BPM",
                fontSize = 14.sp,
                color = SerenColors.textSecondary,
            )

            Spacer(modifier = Modifier.height(12.dp))

            // HRV
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (state.hrv > 0) "${state.hrv.toInt()}" else "--",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.softBlue,
                    )
                    Text("HRV", fontSize = 10.sp, color = SerenColors.textMuted)
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (state.spo2 > 0) "${state.spo2}%" else "--",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.sageGreen,
                    )
                    Text("SpO2", fontSize = 10.sp, color = SerenColors.textMuted)
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (state.isMonitoring) "Monitoring active" else "Swipe back to dashboard",
                fontSize = 10.sp,
                color = SerenColors.textMuted,
                textAlign = TextAlign.Center,
            )
        }
    }
}
