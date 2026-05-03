// @ts-nocheck
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { AuthUser } from '../types';
import { neonAuthService } from './service';

interface NeonAuthContextState {
  user: AuthUser | null;
  loading: boolean;
  operationLoading: boolean;
  initialized: boolean;
}

interface NeonAuthContextActions {
  setOperationLoading: (loading: boolean) => void;
}

type NeonAuthContextType = NeonAuthContextState & NeonAuthContextActions;

const NeonAuthContext = createContext<NeonAuthContextType | undefined>(undefined);

export function NeonAuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<NeonAuthContextState>({
    user: null,
    loading: true,
    operationLoading: false,
    initialized: false,
  });

  const updateState = (updates: Partial<NeonAuthContextState>) =>
    setState(prev => ({ ...prev, ...updates }));

  const setOperationLoading = (loading: boolean) => updateState({ operationLoading: loading });

  useEffect(() => {
    let isMounted = true;
    let sub: any = null;

    (async () => {
      try {
        const user = await neonAuthService.getCurrentUser();
        if (isMounted) updateState({ user, loading: false, initialized: true });

        sub = neonAuthService.onAuthStateChange(authUser => {
          if (isMounted) updateState({ user: authUser });
        });
      } catch (err) {
        console.warn('[NeonAuth] initialization error:', err);
        if (isMounted) updateState({ user: null, loading: false, initialized: true });
      }
    })();

    return () => {
      isMounted = false;
      sub?.unsubscribe?.();
    };
  }, []);

  return (
    <NeonAuthContext.Provider value={{ ...state, setOperationLoading }}>
      {children}
    </NeonAuthContext.Provider>
  );
}

export function useNeonAuthContext(): NeonAuthContextType {
  const ctx = useContext(NeonAuthContext);
  if (!ctx) throw new Error('useNeonAuthContext must be used within NeonAuthProvider');
  return ctx;
}
