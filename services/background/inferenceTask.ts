import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';
import Constants from 'expo-constants';

const BACKGROUND_INFERENCE_TASK = 'BACKGROUND_INFERENCE_TASK';

// Must be defined at module level (top of bundle) before registerTaskAsync is called.
TaskManager.defineTask(BACKGROUND_INFERENCE_TASK, async () => {
  try {
    await runBackgroundInference();
    return 'success' as any;
  } catch (err) {
    console.error('[Background Task] Error:', err);
    return 'error' as any;
  }
});

export async function runBackgroundInference(): Promise<boolean> {
  try {
    console.log('[Background Task] Inference cycle running at:', new Date().toISOString());
    // TODO: Integrate with actual feature extraction when HealthConnect is configured
    // const featureVector = await extractFeaturesFromHealthConnect();
    // const baseline = await getRecentBaseline();
    // const stressResult = predictStress(featureVector);
    // const anxietyResult = predictAnxiety(featureVector, baseline);
    // await savePredictionToDatabase({ ...results, source: 'background' });
    return true;
  } catch (error) {
    console.error('[Background Task] Inference failed:', error);
    return false;
  }
}

export async function registerBackgroundInferenceTask() {
  // Background fetch is not supported in Expo Go — skip silently.
  const isExpoGo = Constants.appOwnership === 'expo';
  if (isExpoGo) {
    console.log('[Background Task] Skipping registration in Expo Go (not supported)');
    return;
  }

  try {
    const isAvailable = await TaskManager.isAvailableAsync();
    if (!isAvailable) {
      console.warn('[Background Task] TaskManager not available on this device');
      return;
    }

    await BackgroundFetch.registerTaskAsync(BACKGROUND_INFERENCE_TASK, {
      minimumInterval: 15 * 60, // 15 minutes (OS minimum on Android/iOS)
      stopOnTerminate: false,
      startOnBoot: true,
    });

    console.log('[Background Task] Successfully registered background inference task');
  } catch (err: any) {
    // Tolerate "already registered" errors on hot reload
    if (err?.message?.includes('already registered')) {
      console.log('[Background Task] Already registered — skipping');
      return;
    }
    console.warn('[Background Task] Failed to register:', err);
  }
}

export async function unregisterBackgroundInferenceTask() {
  try {
    await TaskManager.unregisterTaskAsync(BACKGROUND_INFERENCE_TASK);
    console.log('[Background Task] Successfully unregistered');
  } catch (err) {
    console.warn('[Background Task] Failed to unregister:', err);
  }
}
