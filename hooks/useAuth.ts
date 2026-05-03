/**
 * Seren — Auth Hook
 * Wraps the template auth system for use in the app.
 * Provides login, signup, logout, and current user state.
 */

// @ts-nocheck
import { useNeonAuth } from '@/template/auth/neon/hook';
import type { AuthUser } from '@/template/auth/types';

export function useAuth() {
  const auth = useNeonAuth();
  return {
    ...auth,
    isAuthenticated: !!auth.user,
  };
}

export type { AuthUser };
