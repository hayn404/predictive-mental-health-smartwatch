import { useEffect } from 'react';
import { AlertProvider } from '@/template';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { WellnessProvider } from '@/hooks/useWellness';
import { DraggableFAB } from '@/components/ui/DraggableFAB';
import { NeonAuthProvider as AuthProvider } from '@/template/auth/neon/context';
import { registerBackgroundInferenceTask } from '@/services/background/inferenceTask';

export default function RootLayout() {
  useEffect(() => {
    // Register background inference task on app startup
    registerBackgroundInferenceTask().catch(err => {
      console.error('Failed to register background task:', err);
    });
  }, []);
  return (
    <AlertProvider>
      <AuthProvider>
        <WellnessProvider>
          <SafeAreaProvider>
            <StatusBar style="dark" />
            <Stack screenOptions={{ headerShown: false }}>
              <Stack.Screen name="auth" options={{ animation: 'fade' }} />
              <Stack.Screen name="onboarding" options={{ animation: 'fade' }} />
              <Stack.Screen name="(tabs)" options={{ animation: 'fade' }} />
            </Stack>
            <DraggableFAB />
          </SafeAreaProvider>
        </WellnessProvider>
      </AuthProvider>
    </AlertProvider>
  );
}
