# UI/UX Design Document: Seren Mental Health Monitor

## Design Vision & Brand Identity
- **App Name**: Seren
- **Core Value**: Proactive, non-clinical, privacy-first mental health monitoring.
- **Visual Style**: Calming, minimalist, and trustworthy. Use "Soft UI" or "Glassmorphism" elements.
- **Color Palette**: Sage greens (growth), soft blues (calm), soft violets (spirituality/creativity), and warm neutrals (comfort). Avoid "medical" reds or harsh high-contrast blacks.
- **Typography**: Sans-serif, rounded (e.g., Quicksand or Inter) for accessibility and friendliness.

## Visily AI Master Prompt
Copy and paste this into Visily’s "Generate with AI" feature:
> "Design a high-fidelity mobile application for 'Seren,' an AI-powered mental health monitoring system that connects to a smartwatch. The UI should feel calm and supportive. Key features include a main dashboard showing real-time stress and anxiety scores (0-100), sleep quality metrics, and a large 'Voice Check-in' button. Use cards with rounded corners to display trends. Include a bottom navigation bar with Home, Insights, Check-in, and Settings. The design must emphasize privacy with subtle 'on-device processing' labels. Include data visualizations like line charts for heart rate variability (HRV) and bar charts for weekly mental health trends. Use a color palette of sage greens, soft blues, and soft violets."

## Page-by-Page Requirements

### Page 1: Onboarding & Watch Sync
- **Purpose**: Setup and privacy assurance.
- **Elements**:
  - Value proposition screens (3-step carousel).
  - "Connect Smartwatch" button with Bluetooth scanning animation.
  - Privacy Toggle: "Keep all data on-device" (Checked by default).
  - Permissions request for HR and Sleep data.

### Page 2: Main Dashboard (The "Seren" Home)
- **Purpose**: At-a-glance mental state.
- **Elements**:
  - Header: "Good Morning, [User Name]. You’re doing well."
  - Primary Metric: A large circular gauge showing "Current Stress Level" (e.g., 24/100 - Low).
  - Secondary Cards: Small grid cards for "Sleep Quality (82%)" and "Anxiety Index (Stable)."
  - Action Button: Floating Action Button (FAB) with a microphone icon for "Voice Check-in."
  - Quick Tip: A small banner: "Your HRV is slightly low today. Try a 2-minute breathing exercise."

### Page 3: Voice Check-in (Conversational UI)
- **Purpose**: Qualitative data collection via AI.
- **Elements**:
  - Waveform animation that reacts to voice.
  - AI Text Prompt: "How are you feeling today? I'm listening."
  - Real-time transcription area.
  - "End & Analyze" button.

### Page 4: Insights & Deep Dive
- **Purpose**: Historical trend analysis.
- **Elements**:
  - Tabs for Daily, Weekly, Monthly views.
  - Line Chart: Correlation between Heart Rate and Stress.
  - Heatmap: Sleep consistency over the month.
  - Legend: Explaining what HRV (Heart Rate Variability) means for their mental health.

### Page 5: Recommendations & Resources
- **Purpose**: Supportive, proactive intervention.
- **Elements**:
  - Categorized cards: "Breathing," "Physical Activity," "Journaling."
  - AI-generated suggestion: "Based on your high stress at 3 PM daily, we recommend a short walk then."
  - Emergency "Get Help Now" button (discreetly placed).

### Page 6: Privacy & Settings
- **Purpose**: Data control.
- **Elements**:
  - "Local Storage" status indicator (Green/Active).
  - "Purge All Data" button.
  - "Clinical Screening" section (links to PHQ-9/GAD-7 questionnaires).
  - Watch connection status and battery life.

## User Stories

| ID | User Role | Requirement | Goal/Benefit |
|---|---|---|---|
| US1 | Student | As a student, I want to see my stress levels during exam week. | To know when I need to take a break before burning out. |
| US2 | User | As a user, I want to record a 30-second voice note about my mood. | So the AI can detect emotional nuances my watch might miss. |
| US3 | Privacy-conscious | As a user, I want to know my data isn't being sent to a cloud server. | To feel safe sharing sensitive mental health information. |
| US4 | Proactive user | As a user, I want to receive a notification if my HRV drops significantly. | To engage in a breathing exercise early to prevent an anxiety attack. |
| US5 | Regular user | As a user, I want to view a weekly summary of my sleep and anxiety. | To identify patterns in my lifestyle that affect my well-being. |
