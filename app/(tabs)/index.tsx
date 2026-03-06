import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Colors, FontSize, FontWeight, Spacing, Radius, Shadow } from '@/constants/theme';
import { useHealthData } from '@/hooks/useHealthData';
import { StressGauge } from '@/components/ui/StressGauge';
import { GlassCard } from '@/components/ui/GlassCard';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { health } = useHealthData();

  return (
    <View style={[styles.container, { backgroundColor: '#F3F4F6' }]}>
      <ScrollView
        style={styles.scroll}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.content,
          { paddingTop: insets.top + Spacing.md, paddingBottom: 100 },
        ]}
      >
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View style={styles.appIconContainer}>
              <Image source={require('@/assets/images/logo.png')} style={styles.appIcon} />
            </View>
            <Text style={styles.nameText}>Hi, Alex</Text>
          </View>
          <View style={styles.headerRight}>
            <TouchableOpacity hitSlop={10} style={{ marginRight: 12 }}>
              <MaterialIcons name="settings" size={20} color={Colors.textMuted} />
            </TouchableOpacity>
            <View style={styles.localBadge}>
              <MaterialIcons name="verified-user" size={14} color={Colors.sageGreenDark} />
              <Text style={styles.localText}>LOCAL</Text>
            </View>
          </View>
        </View>

        {/* Greeting */}
        <View style={styles.greetingSection}>
          <Text style={styles.greetingHeading}>Good Morning.</Text>
          <Text style={styles.greetingSub}>You're doing well. Your heart rate is stable.</Text>
        </View>

        {/* Main Stress Gauge */}
        <View style={styles.gaugeCard}>
          <Text style={styles.sectionTitleCenter}>Current Stress Level</Text>
          <View style={styles.gaugeCenter}>
            <StressGauge value={24} size={180} />
          </View>
          <View style={styles.onDevicePill}>
            <MaterialIcons name="verified-user" size={12} color="#35e27e" />
            <Text style={styles.onDevicePillText}>ON-DEVICE PROCESSING</Text>
          </View>
        </View>

        {/* Secondary Metrics Grid */}
        <View style={styles.metricsGrid}>
          <GlassCard variant="default" style={styles.customMetricCard}>
            <View style={styles.metricHeader}>
              <View style={[styles.metricIconBg, { backgroundColor: '#F0F9FF', borderColor: '#E0F2FE' }]}>
                <MaterialIcons name="bedtime" size={18} color={Colors.softBlue} />
              </View>
              <Text style={styles.liveTextSmall}>Live</Text>
            </View>
            <View style={styles.metricContent}>
              <Text style={styles.metricLabel}>Sleep Quality</Text>
              <View style={styles.metricRow}>
                <Text style={styles.metricValue}>82%</Text>
                <Text style={styles.metricSubInfo}>7h 12m</Text>
              </View>
            </View>
          </GlassCard>

          <GlassCard variant="default" style={styles.customMetricCard}>
            <View style={styles.metricHeader}>
              <View style={[styles.metricIconBg, { backgroundColor: '#F0FDF4', borderColor: '#DCFCE7' }]}>
                <MaterialIcons name="show-chart" size={18} color={Colors.sageGreen} />
              </View>
              <Text style={styles.liveTextSmall}>Live</Text>
            </View>
            <View style={styles.metricContent}>
              <Text style={styles.metricLabel}>Anxiety Index</Text>
              <View style={styles.metricRow}>
                <Text style={styles.metricValue}>Stable</Text>
                <Text style={styles.metricSubInfo}>Trend ↓</Text>
              </View>
            </View>
          </GlassCard>
        </View>

        {/* Wellness Summary Tip */}
        <View style={styles.tipBanner}>
          <View style={styles.tipIconWrapper}>
            <MaterialIcons name="info-outline" size={16} color={Colors.sageGreen} />
          </View>
          <View style={styles.tipContent}>
            <Text style={styles.tipText}>
              "Your HRV is slightly low today. Try a 2-minute breathing exercise."
            </Text>
            <TouchableOpacity onPress={() => router.push('/(tabs)/recommendations')} style={styles.tipActionBtn}>
              <Text style={styles.tipAction}>START SESSION</Text>
              <MaterialIcons name="arrow-forward" size={14} color={Colors.sageGreen} />
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      {/* Voice Check-in FAB */}
      <TouchableOpacity
        style={[styles.fab, { bottom: insets.bottom + 16 }]}
        onPress={() => router.push('/(tabs)/checkin')}
        activeOpacity={0.8}
      >
        <MaterialIcons name="mic" size={24} color={Colors.warmWhite} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scroll: {
    flex: 1,
  },
  content: {
    paddingHorizontal: Spacing.lg,
    gap: Spacing.lg,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.sm,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  appIconContainer: {
    width: 36,
    height: 36,
    borderRadius: Radius.md,
    overflow: 'hidden',
    backgroundColor: Colors.warmWhite,
  },
  appIcon: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  nameText: {
    fontSize: FontSize.lg,
    color: Colors.textPrimary,
    fontWeight: FontWeight.bold,
  },
  localBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.sageGreenLight,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.full,
    backgroundColor: Colors.warmWhite,
  },
  localText: {
    fontSize: FontSize.xs,
    color: Colors.sageGreenDark,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  greetingSection: {
    marginVertical: Spacing.sm,
    gap: 2,
  },
  greetingHeading: {
    fontSize: FontSize.xxl,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
  },
  greetingSub: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  gaugeCard: {
    padding: Spacing.xl,
    paddingTop: Spacing.xxl,
    gap: Spacing.md,
    alignItems: 'center',
    marginVertical: Spacing.md,
    borderRadius: 40,
    backgroundColor: Colors.warmWhite,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.03,
    shadowRadius: 15,
    elevation: 2,
  },
  sectionTitleCenter: {
    fontSize: FontSize.md,
    fontWeight: '600',
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  gaugeCenter: {
    marginVertical: Spacing.md,
  },
  onDevicePill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ECFDF5',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: Radius.full,
    gap: 8,
    marginTop: Spacing.xl,
  },
  onDevicePillText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#35e27e',
    letterSpacing: 0.5,
  },
  metricsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: Spacing.md,
    marginBottom: Spacing.sm,
  },
  customMetricCard: {
    flex: 1,
    padding: Spacing.lg,
    borderRadius: Radius.xl,
    backgroundColor: Colors.warmWhite,
    borderWidth: 0,
    shadowColor: Colors.textPrimary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.04,
    shadowRadius: 16,
    elevation: 3,
    gap: Spacing.lg,
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  metricIconBg: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  liveTextSmall: {
    fontSize: 10,
    fontWeight: '600',
    color: Colors.textPrimary,
    marginTop: 4,
  },
  metricContent: {
    gap: 4,
  },
  metricLabel: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  metricRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 6,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    lineHeight: 28,
  },
  metricSubInfo: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
    fontWeight: '500',
    marginBottom: 4,
  },
  tipBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.md,
    padding: Spacing.lg,
    borderRadius: Radius.xl,
    backgroundColor: '#F0FDF4',
    borderWidth: 1,
    borderColor: '#DCFCE7',
    marginVertical: Spacing.md,
  },
  tipIconWrapper: {
    marginTop: 2,
  },
  tipContent: {
    flex: 1,
    gap: Spacing.md,
  },
  tipText: {
    fontSize: FontSize.sm,
    color: Colors.sageGreenDark,
    fontWeight: '500',
    lineHeight: 20,
    fontStyle: 'italic',
  },
  tipActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  tipAction: {
    fontSize: FontSize.xs,
    fontWeight: 'bold',
    color: Colors.sageGreen,
    letterSpacing: 0.5,
  },
  fab: {
    position: 'absolute',
    right: Spacing.lg,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: Colors.violet,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: Colors.warmWhite,
    ...Shadow.glow,
  },
});
