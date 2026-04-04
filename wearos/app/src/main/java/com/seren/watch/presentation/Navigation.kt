package com.seren.watch.presentation

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.wear.compose.navigation.SwipeDismissableNavHost
import androidx.wear.compose.navigation.composable
import androidx.wear.compose.navigation.rememberSwipeDismissableNavController
import com.seren.watch.presentation.screens.AnxietyScreen
import com.seren.watch.presentation.screens.BreathingScreen
import com.seren.watch.presentation.screens.DashboardScreen
import com.seren.watch.presentation.screens.HeartRateScreen
import com.seren.watch.presentation.screens.InsightsScreen
import com.seren.watch.presentation.screens.LocationScreen
import com.seren.watch.presentation.screens.RecommendationsScreen
import com.seren.watch.presentation.screens.SleepScreen
import com.seren.watch.presentation.screens.SunlightScreen

/**
 * Navigation graph for the Seren Wear OS app.
 * Uses swipe-to-dismiss for natural watch navigation.
 */
object Routes {
    const val DASHBOARD = "dashboard"
    const val HEART_RATE = "heart_rate"
    const val SLEEP = "sleep"
    const val BREATHING = "breathing"
    const val SUNLIGHT = "sunlight"
    const val LOCATION = "location"
    const val ANXIETY = "anxiety"
    const val RECOMMENDATIONS = "recommendations"
    const val INSIGHTS = "insights"
}

@Composable
fun SerenWatchNavigation() {
    val navController = rememberSwipeDismissableNavController()
    val viewModel: WellnessViewModel = viewModel()

    SwipeDismissableNavHost(
        navController = navController,
        startDestination = Routes.DASHBOARD,
    ) {
        composable(Routes.DASHBOARD) {
            DashboardScreen(
                viewModel = viewModel,
                onHeartRateTap = { navController.navigate(Routes.HEART_RATE) },
                onSleepTap = { navController.navigate(Routes.SLEEP) },
                onBreathingTap = { navController.navigate(Routes.BREATHING) },
                onSunlightTap = { navController.navigate(Routes.SUNLIGHT) },
                onLocationTap = { navController.navigate(Routes.LOCATION) },
                onAnxietyTap = { navController.navigate(Routes.ANXIETY) },
                onRecommendationsTap = { navController.navigate(Routes.RECOMMENDATIONS) },
                onInsightsTap = { navController.navigate(Routes.INSIGHTS) },
            )
        }

        composable(Routes.HEART_RATE) {
            HeartRateScreen(viewModel = viewModel)
        }

        composable(Routes.SLEEP) {
            SleepScreen(viewModel = viewModel)
        }

        composable(Routes.BREATHING) {
            BreathingScreen()
        }

        composable(Routes.SUNLIGHT) {
            SunlightScreen(viewModel = viewModel)
        }

        composable(Routes.LOCATION) {
            LocationScreen(viewModel = viewModel)
        }

        composable(Routes.ANXIETY) {
            AnxietyScreen(viewModel = viewModel)
        }

        composable(Routes.RECOMMENDATIONS) {
            RecommendationsScreen(
                viewModel = viewModel,
                onBreathingTap = { navController.navigate(Routes.BREATHING) },
            )
        }

        composable(Routes.INSIGHTS) {
            InsightsScreen(viewModel = viewModel)
        }
    }
}
