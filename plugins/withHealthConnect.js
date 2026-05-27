/**
 * plugins/withHealthConnect.js
 *
 * Expo config plugin that injects all required Health Connect
 * permission declarations into AndroidManifest.xml.
 *
 * Health Connect permissions are declared as <uses-permission> entries
 * with the "android.permission.health." prefix. They must also be
 * mirrored as <queries> <package> entries so Android resolves the
 * Health Connect provider at runtime.
 *
 * How to use:
 *   Already referenced in app.json plugins array as "./plugins/withHealthConnect"
 *   No additional configuration needed.
 */

const { withAndroidManifest } = require('@expo/config-plugins');

// All Health Connect record types Seren reads
const HEALTH_CONNECT_READ_PERMISSIONS = [
  'android.permission.health.READ_HEART_RATE',
  'android.permission.health.READ_HEART_RATE_VARIABILITY',
  'android.permission.health.READ_SLEEP',
  'android.permission.health.READ_STEPS',
  'android.permission.health.READ_OXYGEN_SATURATION',
  'android.permission.health.READ_SKIN_TEMPERATURE',
  'android.permission.health.READ_RESPIRATORY_RATE',
  'android.permission.health.READ_EXERCISE',
  'android.permission.health.READ_BODY_TEMPERATURE',
  'android.permission.health.READ_ACTIVE_CALORIES_BURNED',
  'android.permission.health.READ_RESTING_HEART_RATE',
];

// Write permissions — used when Seren logs completed interventions
// back to Health Connect for cross-app visibility
const HEALTH_CONNECT_WRITE_PERMISSIONS = [
  'android.permission.health.WRITE_SLEEP',
  'android.permission.health.WRITE_EXERCISE',
];

const ALL_HC_PERMISSIONS = [
  ...HEALTH_CONNECT_READ_PERMISSIONS,
  ...HEALTH_CONNECT_WRITE_PERMISSIONS,
];

// Health Connect provider package that must be queryable
const HEALTH_CONNECT_PACKAGE = 'com.google.android.apps.healthdata';

/**
 * Adds a <uses-permission android:name="..." /> entry if not already present.
 */
function addPermission(manifest, permissionName) {
  if (!manifest['uses-permission']) {
    manifest['uses-permission'] = [];
  }
  const exists = manifest['uses-permission'].some(
    (p) => p.$?.['android:name'] === permissionName
  );
  if (!exists) {
    manifest['uses-permission'].push({
      $: { 'android:name': permissionName },
    });
  }
}

/**
 * Adds the Health Connect package to <queries> so Android
 * allows Seren to check if Health Connect is installed and
 * open its permission request screen.
 */
function addHealthConnectQuery(manifest) {
  if (!manifest.queries) {
    manifest.queries = [];
  }
  // queries is an array of query blocks; find or create one with <package>
  let queryBlock = manifest.queries.find((q) => q.package);
  if (!queryBlock) {
    queryBlock = { package: [] };
    manifest.queries.push(queryBlock);
  }
  if (!Array.isArray(queryBlock.package)) {
    queryBlock.package = [];
  }
  const exists = queryBlock.package.some(
    (p) => p.$?.['android:name'] === HEALTH_CONNECT_PACKAGE
  );
  if (!exists) {
    queryBlock.package.push({
      $: { 'android:name': HEALTH_CONNECT_PACKAGE },
    });
  }
}

/**
 * Adds the FOREGROUND_SERVICE_HEALTH permission and the
 * foregroundServiceType="health" attribute to the background
 * polling service declaration in <application>.
 *
 * The service entry must already exist in the manifest (added by
 * WorkManager / the health connect library). This function is a
 * no-op if no matching service is found.
 */
function patchHealthForegroundService(application) {
  if (!application.service) return;
  application.service.forEach((service) => {
    const name = service.$?.['android:name'] ?? '';
    // Target WorkManager's SystemForegroundService used for background polling
    if (name.includes('SystemForegroundService') || name.includes('HealthConnect')) {
      if (!service.$) service.$ = {};
      service.$['android:foregroundServiceType'] = 'health';
    }
  });
}

module.exports = withAndroidManifest((config) => {
  const manifest = config.modResults.manifest;

  // 1. Inject all Health Connect permissions
  ALL_HC_PERMISSIONS.forEach((perm) => addPermission(manifest, perm));

  // 2. Add Health Connect package to <queries>
  addHealthConnectQuery(manifest);

  // 3. Patch foreground service type
  if (manifest.application?.[0]) {
    patchHealthForegroundService(manifest.application[0]);
  }

  return config;
});
