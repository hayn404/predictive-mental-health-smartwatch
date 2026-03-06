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

## License

This project is private. For collaboration inquiries, please contact the author.
