package com.seren.watch.presentation.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ShowChart
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
import com.seren.watch.data.AnxietyLevel
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.theme.SerenColors

/**
 * Anxiety index detail screen.
 * Shows the current anxiety level with gauge and sustained status.
 */
@Composable
fun AnxietyScreen(viewModel: WellnessViewModel) {
    val state by viewModel.state.collectAsState()

    val anxietyColor = when (state.anxietyLevel) {
        AnxietyLevel.MINIMAL -> SerenColors.sageGreen
        AnxietyLevel.MILD -> SerenColors.softBlue
        AnxietyLevel.MODERATE -> SerenColors.warning
        AnxietyLevel.SEVERE -> SerenColors.error
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
                Icons.AutoMirrored.Filled.ShowChart,
                contentDescription = "Anxiety",
                tint = anxietyColor,
                modifier = Modifier.size(24.dp),
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Anxiety score arc
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(100.dp),
            ) {
                Canvas(modifier = Modifier.size(100.dp)) {
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
                        color = anxietyColor,
                        startAngle = 150f,
                        sweepAngle = 240f * (state.anxietyIndex / 100f),
                        useCenter = false,
                        topLeft = topLeft,
                        size = arcSize,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )
                }

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "${state.anxietyIndex}",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = SerenColors.textPrimary,
                    )
                    Text(
                        text = state.anxietyLevel.label,
                        fontSize = 10.sp,
                        color = anxietyColor,
                        fontWeight = FontWeight.Medium,
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Anxiety Index",
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = SerenColors.textPrimary,
            )

            if (state.anxietySustained) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Sustained — consider a break",
                    fontSize = 10.sp,
                    color = SerenColors.warning,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
