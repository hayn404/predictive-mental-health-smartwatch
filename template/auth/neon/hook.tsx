// @ts-nocheck
import { AuthContextType, SendOTPResult, AuthResult, LogoutResult, SignUpResult } from '../types';
import { neonAuthService } from './service';
import { useNeonAuthContext } from './context';

export function useNeonAuth(): AuthContextType {
  const ctx = useNeonAuthContext();

  const sendOTP = async (email: string): Promise<SendOTPResult> => {
    ctx.setOperationLoading(true);
    try {
      return await neonAuthService.sendOTP(email);
    } catch {
      return { error: 'Failed to send verification code' };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const verifyOTPAndLogin = async (email: string, otp: string, options?: { password?: string }): Promise<AuthResult> => {
    ctx.setOperationLoading(true);
    try {
      return await neonAuthService.verifyOTPAndLogin(email, otp, options);
    } catch {
      return { error: 'Login failed', user: null };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const signUpWithPassword = async (email: string, password: string, metadata?: Record<string, any>): Promise<SignUpResult> => {
    ctx.setOperationLoading(true);
    try {
      return await neonAuthService.signUpWithPassword(email, password, metadata ?? {});
    } catch {
      return { error: 'Registration failed', user: null };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const signInWithPassword = async (email: string, password: string): Promise<AuthResult> => {
    ctx.setOperationLoading(true);
    try {
      return await neonAuthService.signInWithPassword(email, password);
    } catch {
      return { error: 'Login failed', user: null };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const signInWithGoogle = async (): Promise<AuthResult> => {
    ctx.setOperationLoading(true);
    try {
      return { error: 'Google Sign-In not available', user: null };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const logout = async (): Promise<LogoutResult> => {
    ctx.setOperationLoading(true);
    try {
      return await neonAuthService.logout();
    } catch (err: any) {
      return { error: err?.message ?? 'Logout failed' };
    } finally {
      ctx.setOperationLoading(false);
    }
  };

  const refreshSession = async () => {
    try {
      await neonAuthService.refreshSession();
    } catch (err) {
      console.warn('[NeonAuth] refreshSession error:', err);
    }
  };

  return {
    user: ctx.user,
    loading: ctx.loading,
    operationLoading: ctx.operationLoading,
    initialized: ctx.initialized,
    setOperationLoading: ctx.setOperationLoading,
    sendOTP,
    verifyOTPAndLogin,
    signUpWithPassword,
    signInWithPassword,
    signInWithGoogle,
    logout,
    refreshSession,
  };
}
