# Seren — MVP Features Checklist

| # | Feature | Component | Status | Evidence |
|---|---------|-----------|--------|----------|
| 1 | User Sign Up & Login | Supabase Auth (Backend) | Implemented | Demo video — auth flow |
| 2 | Google OAuth Sign-in | Supabase + expo-auth-session | Implemented | Demo video — login screen |
| 3 | On-Device Stress Prediction (XGBoost) | AI Pipeline (stressModel.ts) | Implemented | Screenshot — stress gauge |
| 4 | Anxiety Index Estimation | AI Pipeline (stressModel.ts) | Implemented | Screenshot — anxiety card |
| 5 | 29-Feature HRV Engineering | AI Pipeline (featureEngineering.ts) | Implemented | Screenshot — HRV trends |
| 6 | Sleep Stage Analysis & Scoring | AI Pipeline (sleepAnalysis.ts) | Implemented | Screenshot — sleep card |
| 7 | Personal Baseline & Anomaly Detection | AI Pipeline (baseline.ts) | Implemented | Demo — anomaly notification |
| 8 | Voice Check-in (Text + Audio) | Mobile App + Whisper API | Implemented | Demo video — check-in flow |
| 9 | Sentiment & Emotion Analysis (VADER + LLM) | AI Pipeline (voiceAnalysis.ts) | Implemented | Screenshot — analysis result |
| 10 | AI Recommendation Engine (20+ interventions) | AI Pipeline (recommendations.ts) | Implemented | Demo — recommendations tab |
| 11 | Guided Breathing Exercises | Mobile + Watch UI | Implemented | Demo — breathing session |
| 12 | Sunlight Exposure Tracking | AI Pipeline (sunlightTracking.ts) | Implemented | Screenshot — sunlight card |
| 13 | Location Diversity Tracking | AI Pipeline (locationTracking.ts) | Implemented | Screenshot — location card |
| 14 | Clinical Screening (PHQ-9 Depression) | Mobile App (screening/phq9.tsx) | Implemented | Demo — PHQ-9 walkthrough |
| 15 | Clinical Screening (GAD-7 Anxiety) | Mobile App (screening/gad7.tsx) | Implemented | Demo — GAD-7 walkthrough |
| 16 | Health Connect Integration (Samsung Watch) | Mobile App (healthConnect.ts) | Implemented | Demo — sensor data flow |
| 17 | Dashboard with Real-Time Metrics | Mobile App (index.tsx) | Implemented | Screenshot — home screen |
| 18 | Historical Trends & Insights | Mobile App (insights.tsx) | Implemented | Screenshot — charts |
| 19 | On-Device SQLite Data Persistence | Database Layer (db.ts) | Implemented | Demo — data persists |
| 20 | Privacy Controls & Data Export/Purge | Mobile App (settings.tsx) | Implemented | Demo — settings screen |
| 21 | Push Notifications (Anomaly & Reminder) | Mobile App (notifications.ts) | Implemented | Demo — HRV drop alert |
| 22 | Wear OS Smartwatch App (8 screens) | Watch App (Kotlin/Compose) | Implemented | Demo — watch app running |
| 23 | Watch: Stress Gauge & HR Display | Watch App (DashboardScreen.kt) | Implemented | Screenshot — watch dashboard |
| 24 | Watch: Guided Breathing with Haptics | Watch App (BreathingScreen.kt) | Implemented | Demo — watch breathing |
| 25 | Onboarding Flow (3 steps) | Mobile App (onboarding/) | Implemented | Demo video — first launch |
