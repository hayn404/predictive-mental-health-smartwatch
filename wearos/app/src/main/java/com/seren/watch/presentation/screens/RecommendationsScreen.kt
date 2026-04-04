package com.seren.watch.presentation.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Air
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.SelfImprovement
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.items
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
import com.seren.watch.data.WatchRecommendation
import com.seren.watch.presentation.WellnessViewModel
import com.seren.watch.presentation.theme.SerenColors

/**
 * Recommendations screen — shows AI-generated suggestions.
 * Each recommendation is a tappable chip with icon, title, and reason.
 */
@Composable
fun RecommendationsScreen(
    viewModel: WellnessViewModel,
    onBreathingTap: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
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
            item {
                Text(
                    text = "For You",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = SerenColors.sageGreen,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            if (state.recommendations.isEmpty()) {
                item {
                    Text(
                        text = "No recommendations right now.\nYou're doing great!",
                        fontSize = 12.sp,
                        color = SerenColors.textSecondary,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 24.dp),
                    )
                }
            } else {
                items(state.recommendations) { rec ->
                    RecommendationChip(
                        recommendation = rec,
                        onClick = {
                            if (rec.category == "breathing") onBreathingTap()
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun RecommendationChip(
    recommendation: WatchRecommendation,
    onClick: () -> Unit,
) {
    val (icon, color) = getCategoryIconAndColor(recommendation.category)

    Chip(
        onClick = onClick,
        colors = ChipDefaults.chipColors(
            backgroundColor = SerenColors.surface,
        ),
        label = {
            Text(
                text = recommendation.title,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        secondaryLabel = {
            Text(
                text = recommendation.triggerReason,
                fontSize = 10.sp,
                color = SerenColors.textMuted,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        },
        icon = {
            Icon(
                icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(20.dp),
            )
        },
        modifier = Modifier.fillMaxWidth(0.95f),
    )
}

private fun getCategoryIconAndColor(category: String): Pair<ImageVector, Color> {
    return when (category) {
        "breathing" -> Icons.Default.Air to SerenColors.softBlue
        "outdoor" -> Icons.Default.WbSunny to SerenColors.warning
        "exploration" -> Icons.Default.Explore to SerenColors.violet
        "physical" -> Icons.Default.SelfImprovement to SerenColors.sageGreen
        else -> Icons.Default.SelfImprovement to SerenColors.textSecondary
    }
}
