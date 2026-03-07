import { AlertProvider } from '@/template';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { WellnessProvider } from '@/hooks/useWellness';

export default function RootLayout() {
  return (
    <AlertProvider>
      <WellnessProvider>
        <SafeAreaProvider>
          <StatusBar style="dark" />
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="onboarding" options={{ animation: 'fade' }} />
            <Stack.Screen name="(tabs)" options={{ animation: 'fade' }} />
          </Stack>
        </SafeAreaProvider>
      </WellnessProvider>
    </AlertProvider>
  );
}
