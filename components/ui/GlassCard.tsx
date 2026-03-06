import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { Colors, Radius, Shadow } from '@/constants/theme';

interface GlassCardProps {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  variant?: 'default' | 'elevated' | 'sage' | 'violet' | 'blue';
}

export function GlassCard({ children, style, variant = 'default' }: GlassCardProps) {
  const variantStyle = variant === 'sage'
    ? styles.sage
    : variant === 'violet'
      ? styles.violet
      : variant === 'blue'
        ? styles.blue
        : variant === 'elevated'
          ? styles.elevated
          : styles.default;

  return (
    <View style={[styles.card, variantStyle, style]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: Radius.lg,
    overflow: 'hidden',
  },
  default: {
    backgroundColor: Colors.warmWhite,
    borderWidth: 1,
    borderColor: Colors.warmGray200,
    ...Shadow.soft,
  },
  elevated: {
    backgroundColor: Colors.warmWhite,
    borderWidth: 1,
    borderColor: Colors.warmGray200,
    ...Shadow.medium,
  },
  sage: {
    backgroundColor: 'rgba(82, 196, 122, 0.10)',
    borderWidth: 1,
    borderColor: 'rgba(82, 196, 122, 0.25)',
    ...Shadow.soft,
  },
  violet: {
    backgroundColor: 'rgba(123, 97, 196, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(123, 97, 196, 0.20)',
    ...Shadow.soft,
  },
  blue: {
    backgroundColor: 'rgba(107, 159, 204, 0.10)',
    borderWidth: 1,
    borderColor: 'rgba(107, 159, 204, 0.25)',
    ...Shadow.soft,
  },
});
