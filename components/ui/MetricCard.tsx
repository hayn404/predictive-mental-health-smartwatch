import React from 'react';
import { View, Text, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from './GlassCard';

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  icon: string;
  trend?: 'up' | 'down' | 'stable';
  trendLabel?: string;
  color: string;
  variant?: 'default' | 'sage' | 'violet' | 'blue';
  style?: StyleProp<ViewStyle>;
}

export function MetricCard({
  label,
  value,
  unit,
  icon,
  trend,
  trendLabel,
  color,
  variant = 'default',
  style,
}: MetricCardProps) {
  const trendIcon = trend === 'up' ? 'trending-up' : trend === 'down' ? 'trending-down' : 'trending-flat';
  const trendColor = trend === 'up' ? Colors.sageGreen : trend === 'down' ? Colors.error : Colors.textMuted;

  return (
    <GlassCard variant={variant} style={[styles.card, style]}>
      <View style={[styles.iconBg, { backgroundColor: color + '20' }]}>
        <MaterialIcons name={icon as any} size={18} color={color} />
      </View>
      <Text style={styles.label} numberOfLines={1}>{label}</Text>
      <View style={styles.valueRow}>
        <Text style={[styles.value, { color }]}>{value}</Text>
        {unit ? <Text style={styles.unit}>{unit}</Text> : null}
      </View>
      {trend ? (
        <View style={styles.trendRow}>
          <MaterialIcons name={trendIcon as any} size={12} color={trendColor} />
          <Text style={[styles.trendText, { color: trendColor }]}>{trendLabel}</Text>
        </View>
      ) : null}
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.md,
    gap: Spacing.xs,
  },
  iconBg: {
    width: 34,
    height: 34,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  label: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 2,
  },
  value: {
    fontSize: FontSize.xl,
    fontWeight: FontWeight.bold,
    lineHeight: 28,
  },
  unit: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
    marginBottom: 2,
  },
  trendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    marginTop: 2,
  },
  trendText: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.medium,
  },
});
