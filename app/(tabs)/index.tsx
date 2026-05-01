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
import { useWellness } from '@/hooks/useWellness';
import { useAuth } from '@/hooks/useAuth';
import { StressGauge } from '@/components/ui/StressGauge';
import { GlassCard } from '@/components/ui/GlassCard';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { stress, anxiety, lastSleep, heartRate, recommendations, sunlightExposure, locationDiversity } = useWellness();
  const { user } = useAuth();
  const displayName = user?.username || user?.email?.split('@')[0] || 'there';

  return (
    <View style={[styles.container, { backgroundColor: Colors.cream }]}>
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
              <Image source={require('@/assets/images/seren-brain.png')} style={styles.appIcon} />
            </View>
            <Text style={styles.nameText}>Hi, {displayName}</Text>
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
          <Text style={styles.greetingHeading}>
            {new Date().getHours() < 12 ? 'Good Morning.' : new Date().getHours() < 17 ? 'Good Afternoon.' : 'Good Evening.'}
          </Text>
          <Text style={styles.greetingSub}>
            {stress.stressLevel === 'low'
              ? `You're doing well. Heart rate is ${heartRate} BPM.`
              : stress.stressLevel === 'moderate'
              ? `Moderate stress detected. Take a moment to breathe.`
              : `Elevated stress detected. Consider a break.`}
          </Text>
        </View>

        {/* Main Stress Gauge */}
        <View style={styles.gaugeCard}>
          <Text style={styles.sectionTitleCenter}>Current Stress Level</Text>
          <View style={styles.gaugeCenter}>
            <StressGauge value={Math.round(stress.stressScore)} size={140} />
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
                <Text style={styles.metricValue}>{lastSleep ? `${Math.round(lastSleep.qualityScore)}%` : '—'}</Text>
                <Text style={styles.metricSubInfo}>{lastSleep ? `${Math.floor(lastSleep.totalSleepMin / 60)}h ${lastSleep.totalSleepMin % 60}m` : 'No data'}</Text>
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
                <Text style={[styles.metricValue, { fontSize: 18 }]} numberOfLines={1} adjustsFontSizeToFit>
                  {anxiety.level.charAt(0).toUpperCase() + anxiety.level.slice(1)}
                </Text>
                <Text style={styles.metricSubInfo}>{anxiety.sustained ? 'Sustained ↑' : 'Trend ↓'}</Text>
              </View>
            </View>
          </GlassCard>
        </View>

        {/* Sunlight Exposure Card */}
        <GlassCard variant="default" style={styles.sunlightCard}>
          <View style={[styles.sunlightLeft, { flex: 1 }]}>
            <View style={[styles.sunlightIconBg, { backgroundColor: '#FEF3C7', borderColor: '#FDE68A' }]}>
              <MaterialIcons name="wb-sunny" size={24} color={Colors.warning} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.metricLabel}>Sunlight Exposure</Text>
              <Text style={styles.metricSubInfo} numberOfLines={1}>
                {sunlightExposure?.isVitaminDWindow ? 'Vitamin D window open!' : 'Daily Goal: 30 mins'}
              </Text>
              {/* Progress bar */}
              <View style={styles.progressBarBg}>
                <View
                  style={[
                    styles.progressBarFill,
                    {
                      width: `${Math.min(100, (sunlightExposure?.goalProgress ?? 0) * 100)}%`,
                      backgroundColor: (sunlightExposure?.goalProgress ?? 0) >= 1 ? Colors.sageGreen : Colors.warning,
                    },
                  ]}
                />
              </View>
            </View>
          </View>
          <View style={styles.sunlightRight}>
            <Text style={styles.sunlightValue}>
              {sunlightExposure ? `${sunlightExposure.totalOutdoorMinutes}m` : '—'}
            </Text>
            <Text style={styles.liveTextSmall}>Today</Text>
          </View>
        </GlassCard>

        {/* Location Diversity Card */}
        <GlassCard variant="default" style={styles.sunlightCard}>
          <View style={[styles.sunlightLeft, { flex: 1 }]}>
            <View style={[styles.sunlightIconBg, { backgroundColor: '#F3F0FF', borderColor: '#E8E0FF' }]}>
              <MaterialIcons name="place" size={24} color={Colors.violet} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.metricLabel}>Location Diversity</Text>
              <Text style={styles.metricSubInfo}>
                {locationDiversity
                  ? `${locationDiversity.uniquePlacesVisited} place${locationDiversity.uniquePlacesVisited !== 1 ? 's' : ''} visited today`
                  : 'Tracking your routine'}
              </Text>
              {locationDiversity?.isMonotonous && (
                <Text style={styles.monotonousHint}>Try visiting somewhere new today</Text>
              )}
            </View>
          </View>
          <View style={styles.sunlightRight}>
            <Text style={styles.sunlightValue}>
              {locationDiversity ? `${locationDiversity.diversityScore}` : '—'}
            </Text>
            <Text style={styles.liveTextSmall}>/100</Text>
          </View>
        </GlassCard>

        {/* Clinical Assessments */}
        <View style={styles.assessmentsSection}>
          <Text style={styles.sectionTitleLeft}>Clinical Assessments</Text>
          <View style={styles.assessmentCards}>
            <TouchableOpacity onPress={() => router.push('/screening/phq9')} activeOpacity={0.8}>
              <GlassCard variant="default" style={styles.assessmentCard}>
                <View style={[styles.assessmentIconBg, { backgroundColor: '#F0F9FF', borderColor: '#E0F2FE' }]}>
                  <MaterialIcons name="assignment" size={24} color={Colors.softBlue} />
                </View>
                <View style={styles.assessmentContent}>
                  <Text style={styles.assessmentLabel}>Depression (PHQ-9)</Text>
                  <Text style={styles.assessmentDesc}>Check your mood and energy levels.</Text>
                </View>
                <MaterialIcons name="chevron-right" size={24} color={Colors.warmGray400} />
              </GlassCard>
            </TouchableOpacity>

            <TouchableOpacity onPress={() => router.push('/screening/gad7')} activeOpacity={0.8}>
              <GlassCard variant="default" style={styles.assessmentCard}>
                <View style={[styles.assessmentIconBg, { backgroundColor: '#FFFBEB', borderColor: '#FEF3C7' }]}>
                  <MaterialIcons name="assignment-late" size={24} color={Colors.warning} />
                </View>
                <View style={styles.assessmentContent}>
                  <Text style={styles.assessmentLabel}>Anxiety (GAD-7)</Text>
                  <Text style={styles.assessmentDesc}>Monitor your stress and worry.</Text>
                </View>
                <MaterialIcons name="chevron-right" size={24} color={Colors.warmGray400} />
              </GlassCard>
            </TouchableOpacity>
          </View>
        </View>

        {/* Wellness Summary Tip */}
        <View style={styles.tipBanner}>
          <View style={styles.tipIconWrapper}>
            <MaterialIcons name="info-outline" size={16} color={Colors.sageGreen} />
          </View>
          <View style={styles.tipContent}>
            <Text style={styles.tipText}>
              {recommendations.length > 0
                ? `"${recommendations[0].triggerReason}"`
                : '"Your vitals look good. Keep it up!"'}
            </Text>
            <TouchableOpacity onPress={() => router.push('/(tabs)/recommendations')} style={styles.tipActionBtn}>
              <Text style={styles.tipAction}>START SESSION</Text>
              <MaterialIcons name="arrow-forward" size={14} color={Colors.sageGreen} />
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
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
    gap: Spacing.md, // compressed from Spacing.lg
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
    padding: Spacing.lg, // compressed from xl
    paddingTop: Spacing.lg, // compressed from xxl
    gap: Spacing.sm, // compressed from md
    alignItems: 'center',
    marginVertical: 4, // compressed from Spacing.md
    borderRadius: 32,
    backgroundColor: Colors.warmWhite,
  },
  sectionTitleCenter: {
    fontSize: FontSize.md,
    fontWeight: '600',
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  sectionTitleLeft: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  assessmentsSection: {
    marginTop: Spacing.sm,
  },
  assessmentCards: {
    gap: Spacing.md,
  },
  assessmentCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    gap: Spacing.md,
    borderRadius: Radius.xl,
    backgroundColor: Colors.warmWhite,
    borderWidth: 0,
  },
  assessmentIconBg: {
    width: 48,
    height: 48,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  assessmentContent: {
    flex: 1,
    gap: 4,
  },
  assessmentLabel: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  assessmentDesc: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
  },
  gaugeCenter: {
    marginVertical: Spacing.xs, // compressed from Spacing.md
  },
  onDevicePill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ECFDF5',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: Radius.full,
    gap: 6,
    marginTop: Spacing.sm, // compressed from Spacing.xl
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
    gap: Spacing.sm, // compressed from md
    marginBottom: 4, // compressed from sm
  },
  customMetricCard: {
    flex: 1,
    padding: Spacing.md, // compressed from lg
    borderRadius: Radius.xl,
    backgroundColor: Colors.warmWhite,
    borderWidth: 0,
    gap: Spacing.md, // compressed from lg
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
  sunlightCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: Spacing.lg,
    borderRadius: Radius.xl,
    backgroundColor: Colors.warmWhite,
    borderWidth: 0,
    marginBottom: Spacing.sm,
  },
  sunlightLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  sunlightIconBg: {
    width: 44,
    height: 44,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  sunlightRight: {
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginLeft: Spacing.sm,
    minWidth: 60,
  },
  sunlightValue: {
    fontSize: 22,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    lineHeight: 26,
  },
  progressBarBg: {
    height: 4,
    borderRadius: 2,
    backgroundColor: '#F3F4F6',
    marginTop: 6,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 2,
  },
  monotonousHint: {
    fontSize: 11,
    color: Colors.violet,
    fontWeight: '600',
    marginTop: 4,
  },
});
