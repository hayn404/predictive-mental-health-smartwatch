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
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Warning
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
 * Location diversity tracking screen.
 * Shows diversity score, unique places visited, and monotony warnings.
 */
@Composable
fun LocationScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()

    val scoreColor = when {
        state.locationDiversityScore >= 50 -> SerenColors.sageGreen
        state.locationDiversityScore >= 20 -> SerenColors.warning
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
                Icons.Default.Place,
                contentDescription = "Location",
                tint = SerenColors.violet,
                modifier = Modifier.size(24.dp),
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Diversity score arc
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(90.dp),
            ) {
                Canvas(modifier = Modifier.size(90.dp)) {
                    val strokeWidth = 8.dp.toPx()
                    val arcSize = Size(size.width - strokeWidth, size.height - strokeWidth)
                    val topLeft = Offset(strokeWidth / 2, strokeWidth / 2)

                    drawArc(
                        color = SerenColors.surfaceLight,
                        startAngle = 150f,
                        sweepAngle = 240f,
                        useCenter = false,
                        topLeft = topLeft,
                        size = arcSize,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )

                    drawArc(
                        color = scoreColor,
                        startAngle = 150f,
                        sweepAngle = 240f * (state.locationDiversityScore / 100f),
                        useCenter = false,
                        topLeft = topLeft,
                        size = arcSize,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "${state.locationDiversityScore}",
                        fontSize = 28.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.textPrimary,
                    )
                    Text(
                        text = "/100",
                        fontSize = 10.sp,
                        color = SerenColors.textSecondary,
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Location Diversity",
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                color = SerenColors.textPrimary,
            )

            Spacer(modifier = Modifier.height(4.dp))

            // Places visited
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
            ) {
                Icon(
                    Icons.Default.Explore,
                    contentDescription = null,
                    tint = SerenColors.textSecondary,
                    modifier = Modifier.size(14.dp),
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "${state.uniquePlacesVisited} place${if (state.uniquePlacesVisited != 1) "s" else ""} today",
                    fontSize = 11.sp,
                    color = SerenColors.textSecondary,
                )
            }

            // Monotonous routine warning
            if (state.isMonotonousRoutine) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                ) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = null,
                        tint = SerenColors.warning,
                        modifier = Modifier.size(12.dp),
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Try somewhere new!",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        color = SerenColors.warning,
                    )
                }
            }
        }
    }
}
