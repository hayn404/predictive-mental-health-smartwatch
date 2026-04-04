/**
 * Seren — Auth Hook
 * Wraps the template auth system for use in the app.
 * Provides login, signup, logout, and current user state.
 */

// @ts-nocheck
import { useCallback } from 'react';
import { useAuthContext } from '@/template/auth/supabase/context';
import { authService } from '@/template/auth/supabase/service';
import type { AuthUser, AuthResult, SignUpResult } from '@/template/auth/types';

export function useAuth() {
  const { user, loading, initialized, operationLoading, setOperationLoading } = useAuthContext();

  const signUpWithPassword = useCallback(async (
    email: string,
    password: string,
    username?: string,
  ): Promise<SignUpResult> => {
    setOperationLoading(true);
    try {
      const result = await authService.signUpWithPassword(email, password, { username });
      return result;
    } finally {
      setOperationLoading(false);
    }
  }, []);

  const signInWithPassword = useCallback(async (
    email: string,
    password: string,
  ): Promise<AuthResult> => {
    setOperationLoading(true);
    try {
      const result = await authService.signInWithPassword(email, password);
      return result;
    } finally {
      setOperationLoading(false);
    }
  }, []);

  const signInWithGoogle = useCallback(async () => {
    setOperationLoading(true);
    try {
      const result = await authService.signInWithGoogle();
      return result;
    } finally {
      setOperationLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setOperationLoading(true);
    try {
      const result = await authService.logout();
      return result;
    } finally {
      setOperationLoading(false);
    }
  }, []);

  return {
    user,
    loading,
    initialized,
    operationLoading,
    isAuthenticated: !!user,
    signUpWithPassword,
    signInWithPassword,
    signInWithGoogle,
    logout,
  };
}

export type { AuthUser };
