/**
 * Optional crash / error reporting (Sentry).
 *
 * DORMANT until `EXPO_PUBLIC_SENTRY_DSN` is set in `.env`:
 *   1. create a project at https://sentry.io (platform: React Native)
 *   2. copy the DSN into `.env` as EXPO_PUBLIC_SENTRY_DSN=...
 *   3. rebuild the dev/app build so the native @sentry module is linked
 *
 * The `@sentry/react-native` module is imported DYNAMICALLY and only when a DSN is
 * present, so a build without Sentry (or without the native module yet) runs fine.
 */
export async function initSentry(): Promise<void> {
  const dsn = process.env.EXPO_PUBLIC_SENTRY_DSN;
  if (!dsn) return; // not configured -> stay dormant
  try {
    const Sentry = await import('@sentry/react-native');
    Sentry.init({
      dsn,
      // Capture a sample of performance traces; tune for production.
      tracesSampleRate: 0.2,
      // Health data stays on-device; never attach PII automatically.
      sendDefaultPii: false,
    });
    console.log('[Seren] Sentry crash reporting initialized');
  } catch (e) {
    console.warn('[Seren] Sentry init skipped:', e);
  }
}
