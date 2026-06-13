import * as TaskManager from 'expo-task-manager';
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

  // Background execution is currently DISABLED (gap S3): the inference body above is a
  // stub, and the native background-fetch module isn't installed. To enable it later:
  //   1. npx expo install expo-background-fetch   (then rebuild the native app)
  //   2. re-add: import * as BackgroundFetch from 'expo-background-fetch'
  //   3. register the task here:
  //        await BackgroundFetch.registerTaskAsync(BACKGROUND_INFERENCE_TASK, {
  //          minimumInterval: 15 * 60, stopOnTerminate: false, startOnBoot: true });
  //   4. implement runBackgroundInference() (extract features -> predict -> persist).
  console.log('[Background Task] Background inference disabled (not yet implemented).');
}

export async function unregisterBackgroundInferenceTask() {
  try {
    await TaskManager.unregisterTaskAsync(BACKGROUND_INFERENCE_TASK);
    console.log('[Background Task] Successfully unregistered');
  } catch (err) {
    console.warn('[Background Task] Failed to unregister:', err);
  }
}
