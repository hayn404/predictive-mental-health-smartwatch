/**
 * Expo Config Plugin — Health Connect Integration
 * Adds required manifest entries for react-native-health-connect:
 *   - Health Connect permissions
 *   - Permission rationale intent filter
 *   - Health Connect package query
 */
const { withAndroidManifest } = require('@expo/config-plugins');

function withHealthConnect(config) {
  return withAndroidManifest(config, (config) => {
    const manifest = config.modResults.manifest;

    // 1. Add <queries> for Health Connect package
    if (!manifest['queries']) {
      manifest['queries'] = [];
    }
    manifest['queries'].push({
      package: [{ $: { 'android:name': 'com.google.android.apps.healthdata' } }],
    });

    // 2. Add intent filter to main activity for permission rationale
    const application = manifest.application?.[0];
    if (application?.activity) {
      const mainActivity = application.activity.find(
        (a) => a.$?.['android:name'] === '.MainActivity'
      );
      if (mainActivity) {
        if (!mainActivity['intent-filter']) {
          mainActivity['intent-filter'] = [];
        }
        // Add Health Connect permission rationale intent
        mainActivity['intent-filter'].push({
          action: [{ $: { 'android:name': 'androidx.health.ACTION_SHOW_PERMISSIONS_RATIONALE' } }],
        });
      }
    }

    // 3. Add a permission rationale activity (required by Health Connect)
    if (application) {
      if (!application.activity) {
        application.activity = [];
      }
      application.activity.push({
        $: {
          'android:name': 'com.seren.app.HealthPermissionRationaleActivity',
          'android:exported': 'true',
        },
        'intent-filter': [
          {
            action: [
              { $: { 'android:name': 'androidx.health.ACTION_SHOW_PERMISSIONS_RATIONALE' } },
            ],
          },
        ],
      });
    }

    return config;
  });
}

module.exports = withHealthConnect;
