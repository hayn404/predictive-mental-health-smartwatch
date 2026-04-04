package com.seren.watch.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.seren.watch.presentation.theme.SerenWatchTheme

/**
 * Main entry point for the Seren Wear OS app.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SerenWatchTheme {
                SerenWatchNavigation()
            }
        }
    }
}
