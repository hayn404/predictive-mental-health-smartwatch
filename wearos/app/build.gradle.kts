plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.seren.watch"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.seren.watch"
        // Wear OS 3+ (Galaxy Watch 4+), minSdk 30 covers all modern Wear OS watches
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    // Compose for Wear OS
    val composeVersion = "1.4.0"
    val wearComposeVersion = "1.4.0"

    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material:material-icons-extended")

    // Wear OS Compose
    implementation("androidx.wear.compose:compose-material:$wearComposeVersion")
    implementation("androidx.wear.compose:compose-foundation:$wearComposeVersion")
    implementation("androidx.wear.compose:compose-navigation:$wearComposeVersion")

    // Wear OS Core
    implementation("androidx.wear:wear:1.3.0")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")

    // Health Services API (on-watch real-time sensor data)
    implementation("androidx.health:health-services-client:1.1.0-alpha05")

    // Wear Data Layer (phone <-> watch sync)
    implementation("com.google.android.gms:play-services-wearable:18.2.0")

    // Tiles API (quick-glance tile on watch face)
    implementation("androidx.wear.tiles:tiles:1.4.1")
    implementation("androidx.wear.tiles:tiles-material:1.4.1")

    // DataStore for local preferences
    implementation("androidx.datastore:datastore-preferences:1.1.2")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.9.0")

    // Debug
    debugImplementation("androidx.compose.ui:ui-tooling")
}
