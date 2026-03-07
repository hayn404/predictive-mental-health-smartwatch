# Welcome to Seren

Seren is a React Native app built with Expo SDK 54, supporting iOS, Android, and Web.

## Requirements

- Node.js 20+
- [Expo Go](https://expo.dev/go) app on your phone (iOS or Android)

## Getting Started

### 1. Install Dependencies

```bash
npm install --legacy-peer-deps
```

> **Note:** `--legacy-peer-deps` is required due to peer dependency conflicts between some packages.

### 2. Run the App

```bash
npx expo start --clear   # first time (clears Metro cache)
npx expo start           # every time after
```

Then scan the **QR code** in the terminal with your phone's camera (iOS) or the Expo Go app (Android).

### Other platforms

```bash
npm run android   # Android emulator
npm run ios       # iOS simulator (Mac only)
npm run web       # Browser
```

### Reset cache if things break

```bash
npm run reset-project
# or
npx expo start --clear
```

## Tech Stack

| | |
|---|---|
| Framework | React Native + Expo SDK 54 |
| Language | TypeScript |
| Navigation | Expo Router v6 |
| React Native | 0.81.5 |
| React | 19.1.0 |
| Backend | Supabase |

## Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## App Pages & Structure

Based on the core user stories and design requirements, the application has the following pages:

1. **Welcome & Vision** (`/app/onboarding/welcome.tsx`): 3-step value proposition carousel introducing Seren as a proactive mental wellness companion.
2. **Device Sync** (`/app/onboarding/sync.tsx`): Smartwatch Bluetooth connection, showing live battery status and connection strength.
3. **Privacy & Permissions** (`/app/onboarding/privacy.tsx`): Health data permissions (HR/Sleep) and the interactive "Keep all data on-device" privacy toggle.
4. **Main Dashboard ("Seren" Home)** (`/app/(tabs)/index.tsx`): Real-time stress levels, sleep quality, anxiety index, and proactive insights/notifications.
5. **Voice Check-in** (`/app/(tabs)/checkin.tsx`): Empathetic conversational UI to record mood notes to capture emotional nuances.
6. **Insights & Deep Dive** (`/app/(tabs)/insights.tsx`): Historical trend analysis for stress, sleep, and HRV.
7. **Recommendations & Resources** (`/app/(tabs)/recommendations.tsx`): Proactive interventions, breathing exercises, and AI suggestions based on daily rhythms.
8. **Privacy, Settings & Clinical Tools** (`/app/(tabs)/settings.tsx`): Data control, purge options, watch status, and links to clinical screening tools like PHQ-9.

## License

This project is private. For collaboration inquiries, please contact the author.
