/**
 * Seren AI — Notification Service
 * ==================================
 * Sends push notifications for anomaly alerts and wellness reminders.
 * Uses expo-notifications for local notifications (no server needed).
 */

import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import { AnomalyFlag } from './baseline';
import { StressPrediction, AnxietyPrediction } from './types';

// ============================================================
// Configuration
// ============================================================

// Configure how notifications appear when app is in foreground
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

// ============================================================
// Permission
// ============================================================

let permissionGranted = false;

export async function requestNotificationPermission(): Promise<boolean> {
  try {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    permissionGranted = finalStatus === 'granted';
    return permissionGranted;
  } catch {
    return false;
  }
}

// ============================================================
// Anomaly Notifications
// ============================================================

// Track last notification time to prevent spam
const lastNotificationTime: Record<string, number> = {};
const COOLDOWN_MS = 30 * 60 * 1000; // 30 minutes between same-type notifications

function shouldNotify(type: string): boolean {
  const now = Date.now();
  const last = lastNotificationTime[type] || 0;
  if (now - last < COOLDOWN_MS) return false;
  lastNotificationTime[type] = now;
  return true;
}

/**
 * Send notifications for detected anomalies.
 */
export async function notifyAnomalies(anomalies: AnomalyFlag[]): Promise<void> {
  if (!permissionGranted) return;

  for (const anomaly of anomalies) {
    const type = `anomaly_${anomaly.metric}`;
    if (!shouldNotify(type)) continue;

    const { title, body } = anomalyToNotification(anomaly);
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: { type: 'anomaly', metric: anomaly.metric },
        categoryIdentifier: 'wellness_alert',
      },
      trigger: null, // Immediately
    });
  }
}

function anomalyToNotification(anomaly: AnomalyFlag): { title: string; body: string } {
  switch (anomaly.metric) {
    case 'rmssd':
      return {
        title: 'Low HRV Detected',
        body: `Your HRV dropped to ${Math.round(anomaly.currentValue)}ms (${anomaly.deviations.toFixed(1)} std below baseline). Consider a breathing exercise.`,
      };
    case 'hrMean':
      return {
        title: anomaly.direction === 'high' ? 'Elevated Heart Rate' : 'Low Heart Rate',
        body: `Heart rate is ${Math.round(anomaly.currentValue)} BPM, ${Math.abs(anomaly.deviations).toFixed(1)} std ${anomaly.direction === 'high' ? 'above' : 'below'} your baseline.`,
      };
    case 'tempMean':
      return {
        title: 'Unusual Temperature',
        body: `Wrist temperature is ${anomaly.currentValue.toFixed(1)}°C, which is unusual for you.`,
      };
    default:
      return {
        title: 'Wellness Alert',
        body: `An unusual pattern was detected in your ${anomaly.metric} readings.`,
      };
  }
}

/**
 * Notify sustained high stress.
 */
export async function notifyHighStress(stress: StressPrediction): Promise<void> {
  if (!permissionGranted) return;
  if (stress.stressScore < 70) return;
  if (!shouldNotify('high_stress')) return;

  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'High Stress Detected',
      body: `Your stress level is at ${Math.round(stress.stressScore)}%. Take a moment to breathe or step away.`,
      data: { type: 'stress_alert' },
    },
    trigger: null,
  });
}

/**
 * Notify sustained anxiety.
 */
export async function notifySustainedAnxiety(anxiety: AnxietyPrediction): Promise<void> {
  if (!permissionGranted) return;
  if (!anxiety.sustained || anxiety.anxietyIndex < 60) return;
  if (!shouldNotify('sustained_anxiety')) return;

  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Sustained Anxiety Detected',
      body: 'Your anxiety levels have been elevated for an extended period. Would you like to try a calming exercise?',
      data: { type: 'anxiety_alert' },
    },
    trigger: null,
  });
}

/**
 * Schedule a daily check-in reminder.
 */
export async function scheduleDailyCheckinReminder(
  hour: number = 20,
  minute: number = 0,
): Promise<void> {
  if (!permissionGranted) return;

  // Cancel existing daily reminders
  await Notifications.cancelAllScheduledNotificationsAsync();

  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Evening Check-in',
      body: 'How was your day? Take a moment to reflect with a quick voice check-in.',
      data: { type: 'daily_checkin' },
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.DAILY,
      hour,
      minute,
    },
  });
}

/**
 * Cancel all scheduled notifications.
 */
export async function cancelAllNotifications(): Promise<void> {
  await Notifications.cancelAllScheduledNotificationsAsync();
}
