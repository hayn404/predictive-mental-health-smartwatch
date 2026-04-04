# Seren — Demo Video Script (2-3 minutes)

## Scene 1: Problem Context (15 seconds)
**[Screen: Title card with Seren logo]**

> "Mental health issues often go undetected until they become severe. Traditional self-reporting is infrequent and unreliable. Seren solves this by continuously monitoring biometric signals from a smartwatch to predict stress, track sleep quality, and deliver personalized interventions — all while keeping your data completely private on your device."

---

## Scene 2: Sign Up & Login (15 seconds)
**[Screen: Phone showing login screen]**

> "Users start by creating an account with email and password, powered by Supabase authentication."

**[Action: Show signup → email confirmation → login]**

> "After signing in, a 3-step onboarding introduces the app's capabilities and connects the smartwatch via Health Connect."

**[Action: Show onboarding slides → watch sync screen]**

---

## Scene 3: Home Dashboard (20 seconds)
**[Screen: Home tab on phone]**

> "The dashboard shows real-time wellness metrics. The central stress gauge displays the XGBoost model's prediction — a score from 0 to 100 computed from 29 HRV features extracted every 5 minutes."

**[Action: Point to stress gauge, sleep card, anxiety card]**

> "Below, we see sleep quality from last night, the anxiety index, sunlight exposure with a vitamin D window indicator, and location diversity — which flags repetitive routines linked to depression."

**[Action: Scroll to sunlight and location cards]**

---

## Scene 4: AI Stress Model (20 seconds)
**[Screen: Code or diagram showing the pipeline]**

> "The stress prediction pipeline works entirely on-device. Raw heart rate and HRV data from the watch are processed into a 29-feature vector — including time-domain, frequency-domain, and non-linear HRV metrics. These feed into an XGBoost model trained on the WESAD stress dataset with 90.1% accuracy. The model runs in pure TypeScript — no cloud inference needed."

---

## Scene 5: Voice Check-in (20 seconds)
**[Screen: Check-in tab]**

> "Users can do a voice or text check-in. The app records audio, transcribes it with Whisper, then analyzes sentiment and emotions using both a VADER-style lexicon and an optional LLM for deeper understanding."

**[Action: Type a check-in message → show analysis result with sentiment, emotions, empathy response]**

> "The system cross-references the emotional analysis with current biometric data — if you say 'I feel fine' but your HRV is dropping, it flags the discrepancy."

---

## Scene 6: Recommendations (15 seconds)
**[Screen: Recommendations tab]**

> "Based on stress levels, sleep quality, sunlight exposure, and location patterns, the AI engine selects from 20+ evidence-based interventions — each with clinical citations. Here it's suggesting a breathing exercise because stress is elevated."

**[Action: Tap a breathing recommendation → show instructions]**

---

## Scene 7: Insights & Clinical Screening (15 seconds)
**[Screen: Insights tab → then PHQ-9 screen]**

> "The insights screen shows weekly trends for stress, sleep, HRV, sunlight, and location diversity."

**[Action: Scroll through charts]**

> "We also implemented validated clinical assessments — PHQ-9 for depression and GAD-7 for anxiety screening."

**[Action: Start PHQ-9, answer a few questions, show result]**

---

## Scene 8: Wear OS Watch App (20 seconds)
**[Screen: Samsung Galaxy Watch running Seren]**

> "The companion Wear OS app runs directly on the Samsung Galaxy Watch. It shows the stress gauge, heart rate, sleep quality, and has a guided breathing exercise with haptic feedback."

**[Action: Scroll through watch dashboard → tap breathing → show animation + vibration]**

> "The watch app is standalone and can be published to the Google Play Store."

---

## Scene 9: Settings & Privacy (10 seconds)
**[Screen: Settings tab]**

> "All data stays on-device in SQLite. Users can export their health data as JSON or purge everything. The only data that leaves the device is the Supabase auth token."

**[Action: Show data export, delete, and sign out options]**

---

## Closing (10 seconds)
**[Screen: Split view — phone + watch]**

> "Seren demonstrates that predictive mental health monitoring can be privacy-first, clinically grounded, and run entirely on consumer hardware. Thank you."

---

## Production Notes
- **Total time:** ~2:40
- **Format:** MP4, screen recording with voice narration
- **Tools:** Use phone screen recorder + watch screen recorder, edit together
- **Each team member presents their section** (AI model, frontend, backend, watch app)
