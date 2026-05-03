// @ts-nocheck
import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouter, usePathname } from 'expo-router';
import { useNeonAuth } from './hook';

const DefaultLoadingScreen = () => (
  <View style={styles.container}>
    <ActivityIndicator size="large" color="#8B5CF6" />
    <Text style={styles.text}>Loading...</Text>
  </View>
);

interface NeonAuthRouterProps {
  children: React.ReactNode;
  loginRoute?: string;
  loadingComponent?: React.ComponentType;
  excludeRoutes?: string[];
}

export function NeonAuthRouter({
  children,
  loginRoute = '/login',
  loadingComponent: LoadingComponent = DefaultLoadingScreen,
  excludeRoutes = [],
}: NeonAuthRouterProps) {
  const { user, loading, initialized } = useNeonAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!initialized || loading) return;

    const isLoginRoute = pathname === loginRoute;
    const isExcluded = excludeRoutes.some(r => pathname.startsWith(r));

    if (!user && !isLoginRoute && !isExcluded) {
      router.push(loginRoute);
    } else if (user && isLoginRoute) {
      router.replace('/');
    }
  }, [user, loading, initialized, pathname, loginRoute, excludeRoutes, router]);

  if (loading || !initialized) return <LoadingComponent />;

  const isLoginRoute = pathname === loginRoute;
  const isExcluded = excludeRoutes.some(r => pathname.startsWith(r));

  if (isLoginRoute || isExcluded || user) return <>{children}</>;
  return <LoadingComponent />;
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FAF9F7' },
  text: { marginTop: 12, fontSize: 14, color: '#9CA3AF' },
});
