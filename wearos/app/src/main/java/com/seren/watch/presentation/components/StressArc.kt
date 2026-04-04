package com.seren.watch.presentation.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.material.Text
import com.seren.watch.data.StressLevel
import com.seren.watch.presentation.theme.SerenColors

/**
 * Circular stress arc gauge for the watch dashboard.
 * Displays the stress score (0-100) as a colored arc.
 */
@Composable
fun StressArc(
    score: Int,
    level: StressLevel,
    modifier: Modifier = Modifier,
    size: Int = 100,
) {
    val sweepAngle = (score / 100f) * 240f  // 240° max arc

    val arcColor = when (level) {
        StressLevel.LOW -> SerenColors.sageGreen
        StressLevel.MODERATE -> SerenColors.softBlue
        StressLevel.ELEVATED -> SerenColors.warning
        StressLevel.HIGH -> SerenColors.error
    }

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier.size(size.dp),
    ) {
        Canvas(modifier = Modifier.size(size.dp)) {
            val strokeWidth = 8.dp.toPx()
            val arcSize = Size(this.size.width - strokeWidth, this.size.height - strokeWidth)
            val topLeft = Offset(strokeWidth / 2, strokeWidth / 2)

            // Background track
            drawArc(
                color = SerenColors.surfaceLight,
                startAngle = 150f,
                sweepAngle = 240f,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
            )

            // Colored progress arc
            drawArc(
                color = arcColor,
                startAngle = 150f,
                sweepAngle = sweepAngle,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
            )
        }

        // Center text
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "$score",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = SerenColors.textPrimary,
            )
            Text(
                text = level.label,
                fontSize = 10.sp,
                color = arcColor,
                fontWeight = FontWeight.Medium,
            )
        }
    }
}
