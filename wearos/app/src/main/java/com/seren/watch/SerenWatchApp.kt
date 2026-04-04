package com.seren.watch

import android.app.Application

/**
 * Seren Watch — Application entry point.
 * Initializes Health Services and data stores.
 */
class SerenWatchApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // Health Services and DataStore are initialized lazily on first access
    }
}
