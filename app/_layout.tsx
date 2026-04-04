import { AlertProvider } from '@/template';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { WellnessProvider } from '@/hooks/useWellness';
import { DraggableFAB } from '@/components/ui/DraggableFAB';
import { AuthProvider } from '@/template/auth/supabase/context';

export default function RootLayout() {
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
