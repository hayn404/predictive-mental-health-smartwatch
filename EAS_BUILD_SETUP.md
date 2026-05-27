# EAS Build Setup — Seren

Complete guide to get a real Health Connect-enabled APK running on a Galaxy Watch + Android phone.

---

## Prerequisites

1. **EAS CLI installed globally**
   ```bash
   npm install -g eas-cli
   ```

2. **Expo account** — create free at https://expo.dev if you don't have one.

3. **Android phone** running Android 9+ with:
   - [Health Connect app](https://play.google.com/store/apps/details?id=com.google.android.apps.healthdata) installed
   - Samsung Health installed and syncing with the Galaxy Watch

---

## Step 1 — Copy the new config files

Place the files from this fix into your project root:

```
your-project/
├── eas.json                          ← replace existing or create new
├── app.json                          ← replace existing
└── plugins/
    └── withHealthConnect.js          ← create new
```

---

## Step 2 — Set your EAS Project ID

1. Run `eas login` and sign in to your Expo account.
2. Run `eas init` inside your project root. This registers the project and prints a `projectId`.
3. Open `app.json` and replace `"YOUR_EAS_PROJECT_ID"` with the actual ID from step 2.

---

## Step 3 — Build the development APK

```bash
eas build --platform android --profile development
```

- This uploads your code to EAS cloud builders and compiles a native APK.
- Build takes approximately 10–15 minutes on first run (subsequent builds are faster due to caching).
- When done, EAS prints a download link for the `.apk` file.

---

## Step 4 — Install the APK

```bash
# Option A: download and install via ADB (phone connected via USB)
adb install path/to/downloaded.apk

# Option B: scan the QR code EAS prints after build — opens download in browser on phone
```

---

## Step 5 — Grant Health Connect permissions

1. Open the **Health Connect** app on the phone.
2. Tap **App permissions** → find **Seren** → grant all permissions:
   - Heart rate ✓
   - Heart rate variability ✓
   - Sleep ✓
   - Steps ✓
   - Skin temperature ✓
   - Oxygen saturation ✓
   - Respiratory rate ✓
3. Make sure **Samsung Health** is connected to Health Connect:
   - Samsung Health → Profile → Connected services → Health Connect → turn on all data types

---

## Step 6 — Verify data is flowing

After wearing the Galaxy Watch for at least 15 minutes:

1. Open Seren — the Home dashboard should show real HR and HRV values (not the placeholder mock values).
2. To confirm via ADB:
   ```bash
   adb shell
   run-as com.seren.mentalhealth
   cat databases/seren.db | sqlite3 /dev/stdin "SELECT COUNT(*) FROM biometric_samples;"
   ```
   A non-zero count confirms real data is being written.

---

## Common Issues

| Problem | Fix |
|---|---|
| `eas: command not found` | Run `npm install -g eas-cli` again; check npm global bin is in PATH |
| Build fails: `minSdkVersion` conflict | Ensure `minSdkVersion: 26` in `app.json` — Health Connect requires Android 9+ |
| Health Connect permissions not appearing | The APK must be the EAS dev build, not Expo Go — Expo Go doesn't support native modules |
| Samsung Health not syncing to Health Connect | Open Samsung Health → Settings → Connected services → Health Connect — enable all toggles |
| `FOREGROUND_SERVICE_HEALTH` permission error | Already handled by `withHealthConnect.js` plugin — rebuild APK if you see this |
| HRV data missing (only HR present) | On Galaxy Watch: Settings → Health → Measurement → Stress measurement → set to "Always on" |

---

## Environment variables (optional)

For the Groq API key (voice check-in LLM), create a `.env.local` file:

```bash
EXPO_PUBLIC_GROQ_API_KEY=your_groq_api_key_here
```

The app functions fully without this — VADER offline fallback handles all voice analysis.
