import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { WatchStatus } from '@/services/mockData';
import { GlassCard } from './GlassCard';

interface WatchPanelProps {
  watchStatus: WatchStatus;
  heartRate: number;
  hrv: number;
}

export function WatchPanel({ watchStatus, heartRate, hrv }: WatchPanelProps) {
  const timeAgo = Math.round((Date.now() - watchStatus.lastSync.getTime()) / 60000);

  return (
    <GlassCard variant="violet" style={styles.card}>
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <MaterialIcons name="watch" size={16} color={Colors.violet} />
          <Text style={styles.title}>Watch</Text>
          <View style={[styles.statusDot, { backgroundColor: watchStatus.connected ? Colors.sageGreen : Colors.error }]} />
          <Text style={[styles.statusText, { color: watchStatus.connected ? Colors.sageGreenDark : Colors.error }]}>
            {watchStatus.connected ? 'Connected' : 'Disconnected'}
          </Text>
        </View>
        <View style={styles.batteryRow}>
          <MaterialIcons
            name={watchStatus.batteryLevel > 20 ? 'battery-full' : 'battery-alert'}
            size={14}
            color={watchStatus.batteryLevel > 20 ? Colors.sageGreen : Colors.warning}
          />
          <Text style={styles.batteryText}>{Math.round(watchStatus.batteryLevel)}%</Text>
        </View>
      </View>

      <View style={styles.metrics}>
        <View style={styles.metric}>
          <MaterialIcons name="favorite" size={14} color={Colors.error} />
          <Text style={styles.metricValue}>{Math.round(heartRate)}</Text>
          <Text style={styles.metricUnit}>bpm</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.metric}>
          <MaterialIcons name="show-chart" size={14} color={Colors.violet} />
          <Text style={styles.metricValue}>{Math.round(hrv)}</Text>
          <Text style={styles.metricUnit}>HRV ms</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.metric}>
          <MaterialIcons name="sync" size={14} color={Colors.softBlue} />
          <Text style={[styles.metricValue, { fontSize: FontSize.sm }]}>{timeAgo}m</Text>
          <Text style={styles.metricUnit}>ago</Text>
        </View>
      </View>

      <Text style={styles.modelText}>{watchStatus.model}</Text>
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  title: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statusText: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.medium,
  },
  batteryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  batteryText: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
    fontWeight: FontWeight.medium,
  },
  metrics: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(155, 142, 196, 0.08)',
    borderRadius: Radius.md,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
  },
  metric: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  metricValue: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
  },
  metricUnit: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  divider: {
    width: 1,
    height: 24,
    backgroundColor: Colors.warmGray300,
  },
  modelText: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    textAlign: 'center',
  },
});
