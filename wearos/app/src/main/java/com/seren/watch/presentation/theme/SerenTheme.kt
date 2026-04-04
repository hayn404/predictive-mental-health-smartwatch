package com.seren.watch.presentation.theme

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.wear.compose.material.Colors
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Typography

// Seren brand colors for dark Wear OS theme
object SerenColors {
    val sageGreen = Color(0xFF35E27E)
    val violet = Color(0xFFA288FC)
    val softBlue = Color(0xFF739EE8)
    val warning = Color(0xFFFFB84D)
    val error = Color(0xFFEF4444)
    val background = Color(0xFF1A1A2E)
    val surface = Color(0xFF242442)
    val surfaceLight = Color(0xFF2E2E52)
    val textPrimary = Color(0xFFFFFFFF)
    val textSecondary = Color(0xFFB0B0C8)
    val textMuted = Color(0xFF6B6B8A)
}

private val SerenWearColors = Colors(
    primary = SerenColors.sageGreen,
    primaryVariant = SerenColors.sageGreen.copy(alpha = 0.7f),
    secondary = SerenColors.violet,
    secondaryVariant = SerenColors.violet.copy(alpha = 0.7f),
    error = SerenColors.error,
    onPrimary = Color.Black,
    onSecondary = Color.White,
    onError = Color.White,
    background = SerenColors.background,
    surface = SerenColors.surface,
    onBackground = SerenColors.textPrimary,
    onSurface = SerenColors.textPrimary,
    onSurfaceVariant = SerenColors.textSecondary,
)

@Composable
fun SerenWatchTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colors = SerenWearColors,
        typography = Typography(),
        content = content,
    )
}
