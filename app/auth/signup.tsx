import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  ScrollView,
  Image,
} from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { useAuth } from '@/hooks/useAuth';

export default function SignUpScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { signUpWithPassword, operationLoading } = useAuth();

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSignUp = async () => {
    if (!username.trim() || !email.trim() || !password.trim()) {
      setError('Please fill in all fields');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setError('');
    const result = await signUpWithPassword(email.trim(), password, username.trim());

    if (result.error) {
      setError(result.error);
    } else if (result.needsEmailConfirmation) {
      setSuccess(true);
    }
    // If user is returned directly, auth state change will trigger navigation
  };

  if (success) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center', padding: Spacing.xl }]}>
        <View style={styles.successIcon}>
          <MaterialIcons name="mark-email-read" size={48} color={Colors.sageGreen} />
        </View>
        <Text style={styles.successTitle}>Check your email</Text>
        <Text style={styles.successText}>
          We sent a confirmation link to {email}. Please verify your email to continue.
        </Text>
        <TouchableOpacity
          style={[styles.primaryButton, { marginTop: Spacing.xl, width: '100%' }]}
          onPress={() => router.replace('/auth/login')}
        >
          <Text style={styles.primaryButtonText}>Go to Login</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        contentContainerStyle={[
          styles.content,
          { paddingTop: insets.top + 40, paddingBottom: insets.bottom + 40 },
        ]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Logo */}
        <View style={styles.logoSection}>
          <View style={styles.logoContainer}>
            <Image source={require('@/assets/images/logo.png')} style={styles.logo} />
          </View>
          <Text style={styles.appName}>Seren</Text>
        </View>

        {/* Form */}
        <View style={styles.form}>
          <Text style={styles.heading}>Create your account</Text>
          <Text style={styles.subtitle}>Start your wellness journey today</Text>

          {error ? (
            <View style={styles.errorBox}>
              <MaterialIcons name="error-outline" size={16} color={Colors.error} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Username</Text>
            <View style={styles.inputContainer}>
              <MaterialIcons name="person-outline" size={18} color={Colors.textMuted} />
              <TextInput
                style={styles.input}
                placeholder="Choose a username"
                placeholderTextColor={Colors.textMuted}
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
                autoComplete="username"
              />
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Email</Text>
            <View style={styles.inputContainer}>
              <MaterialIcons name="mail-outline" size={18} color={Colors.textMuted} />
              <TextInput
                style={styles.input}
                placeholder="you@example.com"
                placeholderTextColor={Colors.textMuted}
                value={email}
                onChangeText={setEmail}
                autoCapitalize="none"
                keyboardType="email-address"
                autoComplete="email"
              />
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Password</Text>
            <View style={styles.inputContainer}>
              <MaterialIcons name="lock-outline" size={18} color={Colors.textMuted} />
              <TextInput
                style={styles.input}
                placeholder="At least 6 characters"
                placeholderTextColor={Colors.textMuted}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                autoComplete="new-password"
              />
              <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
                <MaterialIcons
                  name={showPassword ? 'visibility' : 'visibility-off'}
                  size={18}
                  color={Colors.textMuted}
                />
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Confirm Password</Text>
            <View style={styles.inputContainer}>
              <MaterialIcons name="lock-outline" size={18} color={Colors.textMuted} />
              <TextInput
                style={styles.input}
                placeholder="Re-enter your password"
                placeholderTextColor={Colors.textMuted}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showPassword}
              />
            </View>
          </View>

          <TouchableOpacity
            style={[styles.primaryButton, operationLoading && styles.buttonDisabled]}
            onPress={handleSignUp}
            disabled={operationLoading}
          >
            {operationLoading ? (
              <ActivityIndicator color={Colors.warmWhite} />
            ) : (
              <Text style={styles.primaryButtonText}>Create Account</Text>
            )}
          </TouchableOpacity>

          {/* Privacy note */}
          <Text style={styles.privacyNote}>
            Your health data stays on your device. We only store your account credentials securely.
          </Text>

          {/* Login Link */}
          <View style={styles.loginRow}>
            <Text style={styles.loginText}>Already have an account?</Text>
            <TouchableOpacity onPress={() => router.replace('/auth/login')}>
              <Text style={styles.loginLink}>Sign In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.cream },
  content: { paddingHorizontal: Spacing.xl },
  logoSection: { alignItems: 'center', marginBottom: 32 },
  logoContainer: { width: 60, height: 60, borderRadius: 16, overflow: 'hidden', backgroundColor: Colors.warmWhite, marginBottom: Spacing.sm },
  logo: { width: '100%', height: '100%', resizeMode: 'contain' },
  appName: { fontSize: 24, fontWeight: '800', color: Colors.textPrimary },
  form: { gap: Spacing.md },
  heading: { fontSize: FontSize.xl, fontWeight: FontWeight.bold, color: Colors.textPrimary },
  subtitle: { fontSize: FontSize.sm, color: Colors.textSecondary, marginTop: -8 },
  errorBox: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm, backgroundColor: '#FEF2F2', padding: Spacing.md, borderRadius: Radius.md, borderWidth: 1, borderColor: '#FECACA' },
  errorText: { flex: 1, fontSize: FontSize.sm, color: Colors.error },
  inputGroup: { gap: 6 },
  label: { fontSize: FontSize.sm, fontWeight: '600', color: Colors.textPrimary },
  inputContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.warmWhite, borderWidth: 1, borderColor: '#E2E8F0', borderRadius: Radius.md, paddingHorizontal: Spacing.md, height: 48, gap: Spacing.sm },
  input: { flex: 1, fontSize: FontSize.md, color: Colors.textPrimary },
  primaryButton: { backgroundColor: Colors.violet, paddingVertical: 14, borderRadius: Radius.md, alignItems: 'center', marginTop: Spacing.sm },
  buttonDisabled: { opacity: 0.6 },
  primaryButtonText: { fontSize: FontSize.md, fontWeight: FontWeight.bold, color: Colors.warmWhite },
  privacyNote: { fontSize: 11, color: Colors.textMuted, textAlign: 'center', lineHeight: 16, marginTop: Spacing.sm },
  loginRow: { flexDirection: 'row', justifyContent: 'center', gap: 6, marginTop: Spacing.md },
  loginText: { fontSize: FontSize.sm, color: Colors.textSecondary },
  loginLink: { fontSize: FontSize.sm, fontWeight: FontWeight.bold, color: Colors.violet },
  successIcon: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#F0FDF4', alignItems: 'center', justifyContent: 'center', marginBottom: Spacing.lg },
  successTitle: { fontSize: FontSize.xl, fontWeight: FontWeight.bold, color: Colors.textPrimary, marginBottom: Spacing.sm },
  successText: { fontSize: FontSize.md, color: Colors.textSecondary, textAlign: 'center', lineHeight: 22 },
});
