/**
 * Thin wrapper around the WearableDepressionModule native Android module.
 * Pushes the phone-computed depression risk to the paired Wear OS watch.
 * No-ops silently on non-Android platforms and when no watch is paired.
 */
import { NativeModules, Platform } from 'react-native';
import type { DepressionPrediction } from './ai/types';

const { WearableDepressionModule } = NativeModules;

export async function sendDepressionToWatch(pred: DepressionPrediction): Promise<void> {
  if (Platform.OS !== 'android' || !WearableDepressionModule) return;
  try {
    await WearableDepressionModule.sendDepressionToWatch(
      pred.riskScore,
      pred.riskLevel,
      pred.probability,
    );
  } catch {
    // Non-fatal: watch display is supplementary
  }
}
