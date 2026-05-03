import { useRouter } from 'expo-router';
import { View, ActivityIndicator } from 'react-native';
import { Colors } from '@/constants/theme';
import { useAuth } from '@/hooks/useAuth';
import { useEffect } from 'react';

export default function Index() {
  const { isAuthenticated, loading, initialized } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!initialized || loading) return;
    if (!isAuthenticated) {
      router.replace('/auth/login' as any);
    } else {
      router.replace('/onboarding/welcome' as any);
    }
  }, [initialized, loading, isAuthenticated, router]);

  // Show loading while checking auth state
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: Colors.cream }}>
      <ActivityIndicator size="large" color={Colors.violet} />
    </View>
  );
}
