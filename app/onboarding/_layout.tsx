import { Stack } from 'expo-router';

export default function OnboardingLayout() {
    return (
        <Stack screenOptions={{ headerShown: false, animation: 'fade' }}>
            <Stack.Screen name="welcome" />
            <Stack.Screen name="sync" />
            <Stack.Screen name="privacy" />
        </Stack>
    );
}
