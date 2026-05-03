// @ts-nocheck
import { neon } from '@neondatabase/serverless';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Crypto from 'expo-crypto';
import { AuthUser, SendOTPOptions } from '../types';

// WARNING: In production, move this to a server-side API and do not expose DB credentials in the client bundle.
const DATABASE_URL =
  'postgresql://neondb_owner:npg_iLGRXxrD7tl5@ep-frosty-mode-amkt0rpn-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require';

const sql = neon(DATABASE_URL);

const SESSION_KEY = '@seren_neon_session';
const SALT = 'seren_2024_auth_salt';

interface NeonSession {
  userId: string;
  token: string;
  expiresAt: number;
}

async function hashPassword(password: string): Promise<string> {
  return Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    password + SALT,
  );
}

async function initSchema() {
  await sql`
    CREATE TABLE IF NOT EXISTS seren_users (
      id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      email       TEXT UNIQUE NOT NULL,
      username    TEXT,
      password_hash TEXT,
      created_at  TIMESTAMPTZ DEFAULT NOW(),
      updated_at  TIMESTAMPTZ DEFAULT NOW()
    )
  `;
  await sql`
    CREATE TABLE IF NOT EXISTS seren_sessions (
      id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id     UUID REFERENCES seren_users(id) ON DELETE CASCADE,
      token       TEXT UNIQUE NOT NULL,
      expires_at  TIMESTAMPTZ NOT NULL,
      created_at  TIMESTAMPTZ DEFAULT NOW()
    )
  `;
}

let schemaReady = false;
async function ensureSchema() {
  if (schemaReady) return;
  await initSchema();
  schemaReady = true;
}

function mapRow(row: any): AuthUser {
  return {
    id: row.id,
    email: row.email,
    username: row.username ?? row.email.split('@')[0],
    created_at: row.created_at instanceof Date ? row.created_at.toISOString() : row.created_at,
    updated_at: row.updated_at instanceof Date ? row.updated_at.toISOString() : row.updated_at,
  };
}

async function createSession(userId: string): Promise<void> {
  const token = `seren_${Date.now()}_${Math.random().toString(36).slice(2, 18)}`;
  const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

  await sql`
    INSERT INTO seren_sessions (user_id, token, expires_at)
    VALUES (${userId}, ${token}, ${expiresAt.toISOString()})
  `;

  const session: NeonSession = { userId, token, expiresAt: expiresAt.getTime() };
  await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

// ─────────────────────────────────────────────────────────────────────────────

export class NeonAuthService {
  async signUpWithPassword(email: string, password: string, metadata: Record<string, any> = {}) {
    try {
      await ensureSchema();
      const hash = await hashPassword(password);
      const username = metadata.username || email.split('@')[0];

      const rows = await sql`
        INSERT INTO seren_users (email, username, password_hash)
        VALUES (${email}, ${username}, ${hash})
        RETURNING id, email, username, created_at, updated_at
      `;

      await createSession(rows[0].id);
      return { user: mapRow(rows[0]) };
    } catch (err: any) {
      if (err?.message?.includes('unique') || err?.message?.includes('duplicate')) {
        return { error: 'An account with this email already exists', errorType: 'business' as const };
      }
      console.warn('[NeonAuth] signUpWithPassword error:', err);
      return { error: 'Sign up failed. Please try again.', errorType: 'unknown' as const };
    }
  }

  async signInWithPassword(email: string, password: string) {
    try {
      await ensureSchema();
      const hash = await hashPassword(password);

      const rows = await sql`
        SELECT id, email, username, created_at, updated_at
        FROM seren_users
        WHERE email = ${email} AND password_hash = ${hash}
      `;

      if (rows.length === 0) {
        const exists = await sql`SELECT id FROM seren_users WHERE email = ${email}`;
        if (exists.length === 0) {
          return { error: 'No account found with this email', user: null, errorType: 'business' as const };
        }
        return { error: 'Incorrect password', user: null, errorType: 'business' as const };
      }

      await createSession(rows[0].id);
      return { user: mapRow(rows[0]) };
    } catch (err) {
      console.warn('[NeonAuth] signInWithPassword error:', err);
      return { error: 'Sign in failed. Please try again.', user: null, errorType: 'unknown' as const };
    }
  }

  async sendOTP(email: string, _options: SendOTPOptions = {}) {
    // OTP email delivery not implemented — treated as no-op
    return {};
  }

  async verifyOTPAndLogin(
    email: string,
    _otp: string,
    options?: { password?: string; metadata?: Record<string, any> },
  ) {
    try {
      await ensureSchema();

      let rows = await sql`
        SELECT id, email, username, created_at, updated_at FROM seren_users WHERE email = ${email}
      `;

      if (rows.length === 0) {
        const username = options?.metadata?.username || email.split('@')[0];
        const hash = options?.password ? await hashPassword(options.password) : null;
        rows = await sql`
          INSERT INTO seren_users (email, username, password_hash)
          VALUES (${email}, ${username}, ${hash})
          RETURNING id, email, username, created_at, updated_at
        `;
      }

      await createSession(rows[0].id);
      return { user: mapRow(rows[0]) };
    } catch (err) {
      console.warn('[NeonAuth] verifyOTPAndLogin error:', err);
      return { error: 'Login failed. Please try again.', user: null, errorType: 'unknown' as const };
    }
  }

  async logout() {
    try {
      const raw = await AsyncStorage.getItem(SESSION_KEY);
      if (raw) {
        const session: NeonSession = JSON.parse(raw);
        await sql`DELETE FROM seren_sessions WHERE token = ${session.token}`;
      }
    } catch (err) {
      console.warn('[NeonAuth] logout DB cleanup error:', err);
    } finally {
      await AsyncStorage.removeItem(SESSION_KEY);
    }
    return {};
  }

  async getCurrentUser(): Promise<AuthUser | null> {
    try {
      const raw = await AsyncStorage.getItem(SESSION_KEY);
      if (!raw) return null;

      const session: NeonSession = JSON.parse(raw);
      if (session.expiresAt < Date.now()) {
        await AsyncStorage.removeItem(SESSION_KEY);
        return null;
      }

      const rows = await sql`
        SELECT u.id, u.email, u.username, u.created_at, u.updated_at
        FROM seren_users u
        JOIN seren_sessions s ON s.user_id = u.id
        WHERE s.token = ${session.token} AND s.expires_at > NOW()
      `;

      if (rows.length === 0) {
        await AsyncStorage.removeItem(SESSION_KEY);
        return null;
      }

      return mapRow(rows[0]);
    } catch (err) {
      console.warn('[NeonAuth] getCurrentUser error:', err);
      return null;
    }
  }

  async refreshSession() {
    try {
      const raw = await AsyncStorage.getItem(SESSION_KEY);
      if (!raw) return;

      const session: NeonSession = JSON.parse(raw);
      const newExpiry = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

      await sql`
        UPDATE seren_sessions SET expires_at = ${newExpiry.toISOString()}
        WHERE token = ${session.token}
      `;

      session.expiresAt = newExpiry.getTime();
      await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(session));
    } catch (err) {
      console.warn('[NeonAuth] refreshSession error:', err);
    }
  }

  onAuthStateChange(callback: (user: AuthUser | null) => void) {
    let intervalId: NodeJS.Timeout;
    let lastUserId: string | null | undefined = undefined;

    const check = async () => {
      try {
        const user = await this.getCurrentUser();
        const newId = user?.id ?? null;
        if (newId !== lastUserId) {
          lastUserId = newId;
          callback(user);
        }
      } catch {}
    };

    check();
    intervalId = setInterval(check, 5000);
    return { unsubscribe: () => clearInterval(intervalId) };
  }
}

export const neonAuthService = new NeonAuthService();
