import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';

// Configure how notifications appear when the app is in the foreground
Notifications.setNotificationHandler({
    handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: false,
        shouldShowBanner: true,
        shouldShowList: true,
    }),
});

/**
 * Requests push notification permissions from the user.
 * Best called during the onboarding flow.
 */
export async function requestNotificationPermissions() {
    if (Platform.OS === 'android') {
        await Notifications.setNotificationChannelAsync('alerts', {
            name: 'Proactive Alerts',
            importance: Notifications.AndroidImportance.HIGH,
            vibrationPattern: [0, 250, 250, 250],
            lightColor: '#8B5CF6',
        });
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
    }

    return finalStatus === 'granted';
}

/**
 * Schedules a local mock push notification to trigger 5 seconds in the future
 * to simulate a proactive background HRV drop alert.
 */
export async function scheduleMockHRVDropNotification() {
    const hasPermission = await requestNotificationPermissions();
    if (!hasPermission) return false;

    await Notifications.scheduleNotificationAsync({
        content: {
            title: 'Seren - Proactive Alert',
            body: 'We noticed a sudden drop in your HRV. Taking a 2-minute breathing break can help regulate your nervous system.',
            data: { type: 'hrv_drop', recommendedAction: 'breathe' },
            sound: true,
            badge: 1,
        },
        trigger: {
            type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
            seconds: 5,
        },
    });

    return true;
}
