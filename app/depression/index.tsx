import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { useWellness } from '@/hooks/useWellness';
import { GlassCard } from '@/components/ui/GlassCard';
import type { DepressionRiskLevel } from '@/services/ai/types';

const RISK_META: Record<DepressionRiskLevel, { color: string; bg: string; icon: string; plain: string }> = {
  minimal: {
    color: '#35e27e',
    bg:    '#E8FBF2',
    icon:  'check-circle',
    plain: 'Your daily activity patterns look healthy and consistent with good mood regulation.',
  },
  mild: {
    color: '#9B8EC4',
    bg:    '#F0EDF9',
    icon:  'info',
    plain: 'Minor irregularities in your routine have been detected. Small changes like regular walks or consistent sleep times can help.',
  },
  moderate: {
    color: '#E8A87C',
    bg:    '#FDF4EC',
    icon:  'warning',
    plain: 'Your activity patterns show signs associated with low mood — disrupted sleep proxy, low daytime movement, or reduced variety. Consider a PHQ-9 check-in.',
  },
  high: {
    color: '#C4897B',
    bg:    '#FAEDEA',
    icon:  'error-outline',
    plain: 'Significant disruption in your daily activity rhythm has been detected. Please consider the PHQ-9 screening below and speak with a healthcare professional if needed.',
  },
};

const FEATURE_NICE: Record<string, string> = {
  mean_activity:        'Average activity',
  std_activity:         'Activity variability',
  median_activity:      'Typical activity',
  max_activity:         'Peak activity',
  inactive_minutes:     'Sedentary time',
  active_minutes:       'Active time',
  very_active_minutes:  'High-intensity time',
  day_mean:             'Daytime activity',
  night_mean:           'Night-time movement',
  day_night_ratio:      'Day/night rhythm',
  longest_rest_period:  'Longest rest stretch',
  rest_period_count:    'Rest intervals',
  activity_entropy:     'Routine variety',
  activity_cv:          'Activity consistency',
  p10:                  'Low-activity baseline',
  p90:                  'High-activity peaks',
};

export default function DepressionScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { depression } = useWellness();

  const meta = depression ? RISK_META[depression.riskLevel] : RISK_META.minimal;

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} hitSlop={12} style={styles.backBtn}>
          <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Mood Risk</Text>
        <View style={styles.pill}>
          <MaterialIcons name="verified-user" size={11} color={Colors.sageGreenDark} />
          <Text style={styles.pillText}>ON-DEVICE</Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 32 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Risk level hero */}
        <View style={[styles.heroCard, { backgroundColor: meta.bg, borderColor: meta.color + '44' }]}>
          <View style={styles.heroIconRow}>
            <MaterialIcons name={meta.icon as any} size={32} color={meta.color} />
            <View>
              {depression ? (
                <>
                  <Text style={[styles.heroLevel, { color: meta.color }]}>
                    {depression.riskLevel.charAt(0).toUpperCase() + depression.riskLevel.slice(1)} Risk
                  </Text>
                  <Text style={styles.heroScore}>{depression.riskScore} / 100</Text>
                </>
              ) : (
                <Text style={styles.heroLevel}>Gathering data…</Text>
              )}
            </View>
          </View>
          <Text style={styles.heroPlain}>{meta.plain}</Text>
          {depression?.dataQuality === 'mock' && (
            <Text style={styles.mockNote}>Using simulated data — connect your watch to get real readings.</Text>
          )}
        </View>

        {/* Risk score bar */}
        {depression && (
          <GlassCard variant="default" style={styles.section}>
            <Text style={styles.sectionTitle}>Risk Score</Text>
            <View style={styles.barBg}>
              <View style={[
                styles.barFill,
                { width: `${depression.riskScore}%` as any, backgroundColor: meta.color },
              ]} />
              {/* Threshold marker at 40 */}
              <View style={[styles.thresholdMark, { left: '40%' as any }]} />
            </View>
            <View style={styles.barLegend}>
              <Text style={styles.barLabel}>Minimal</Text>
              <Text style={[styles.barLabel, { color: '#9B8EC4' }]}>Mild</Text>
              <Text style={[styles.barLabel, { color: '#E8A87C' }]}>Moderate</Text>
              <Text style={[styles.barLabel, { color: '#C4897B' }]}>High</Text>
            </View>
          </GlassCard>
        )}

        {/* Top contributing signals */}
        {depression && depression.topContributors.length > 0 && (
          <GlassCard variant="default" style={styles.section}>
            <Text style={styles.sectionTitle}>Top Signals</Text>
            <Text style={styles.sectionSub}>Features that most influenced today's reading</Text>
            {depression.topContributors.map((c, i) => (
              <View key={i} style={styles.contributorRow}>
                <View style={[styles.contributorDot, { backgroundColor: meta.color }]} />
                <View style={styles.contributorText}>
                  <Text style={styles.contributorLabel}>
                    {FEATURE_NICE[c.feature] ?? c.feature}
                  </Text>
                  <Text style={styles.contributorDesc}>{c.label}</Text>
                </View>
                <View style={styles.contributorBar}>
                  <View style={[
                    styles.contributorBarFill,
                    { width: `${Math.round(c.impact * 100)}%` as any, backgroundColor: meta.color + '88' },
                  ]} />
                </View>
              </View>
            ))}
          </GlassCard>
        )}

        {/* Data info */}
        {depression && (
          <GlassCard variant="default" style={styles.section}>
            <Text style={styles.sectionTitle}>About This Reading</Text>
            <View style={styles.infoRow}>
              <MaterialIcons name="today" size={16} color={Colors.textMuted} />
              <Text style={styles.infoText}>
                Based on {depression.daysOfData} day{depression.daysOfData !== 1 ? 's' : ''} of activity data
              </Text>
            </View>
            <View style={styles.infoRow}>
              <MaterialIcons name="timeline" size={16} color={Colors.textMuted} />
              <Text style={styles.infoText}>Updated once daily from step + activity data</Text>
            </View>
            <View style={styles.infoRow}>
              <MaterialIcons name="lock" size={16} color={Colors.textMuted} />
              <Text style={styles.infoText}>Processed entirely on your device</Text>
            </View>
          </GlassCard>
        )}

        {/* Non-clinical disclaimer */}
        <View style={styles.disclaimer}>
          <MaterialIcons name="info-outline" size={16} color={Colors.textMuted} />
          <Text style={styles.disclaimerText}>
            This is a <Text style={{ fontWeight: '700' }}>wellness screening signal</Text>, not a clinical diagnosis.
            It uses wrist actigraphy patterns from the Depresjon research dataset and is intended to complement —
            not replace — professional mental health care.
          </Text>
        </View>

        {/* PHQ-9 cross-reference */}
        <TouchableOpacity
          style={styles.phqBanner}
          onPress={() => router.push('/screening/phq9' as any)}
          activeOpacity={0.8}
        >
          <View style={styles.phqLeft}>
            <MaterialIcons name="assignment" size={22} color={Colors.softBlue} />
            <View>
              <Text style={styles.phqTitle}>Take the PHQ-9</Text>
              <Text style={styles.phqSub}>Clinical depression screening questionnaire</Text>
            </View>
          </View>
          <MaterialIcons name="chevron-right" size={22} color={Colors.textMuted} />
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.cream },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    gap: Spacing.sm,
  },
  backBtn: { padding: 4 },
  headerTitle: { flex: 1, fontSize: FontSize.lg, fontWeight: FontWeight.bold, color: Colors.textPrimary },
  pill: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    borderWidth: 1, borderColor: Colors.sageGreenLight,
    paddingHorizontal: 8, paddingVertical: 4, borderRadius: Radius.full,
    backgroundColor: Colors.warmWhite,
  },
  pillText: { fontSize: 9, color: Colors.sageGreenDark, fontWeight: '700', letterSpacing: 0.5 },
  content: { paddingHorizontal: Spacing.lg, gap: Spacing.md },
  heroCard: {
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    borderWidth: 1,
    gap: Spacing.sm,
  },
  heroIconRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md },
  heroLevel: { fontSize: FontSize.xl, fontWeight: FontWeight.bold },
  heroScore: { fontSize: FontSize.sm, color: Colors.textSecondary, marginTop: 2 },
  heroPlain: { fontSize: FontSize.sm, color: Colors.textSecondary, lineHeight: 20 },
  mockNote: { fontSize: FontSize.xs, color: Colors.textMuted, fontStyle: 'italic', marginTop: 4 },
  section: {
    borderRadius: Radius.lg,
    padding: Spacing.md,
    backgroundColor: Colors.warmWhite,
    borderWidth: 0,
    gap: Spacing.sm,
  },
  sectionTitle: { fontSize: FontSize.md, fontWeight: FontWeight.bold, color: Colors.textPrimary },
  sectionSub: { fontSize: FontSize.xs, color: Colors.textMuted, marginTop: -4 },
  barBg: {
    height: 10, borderRadius: 5, backgroundColor: Colors.warmGray100,
    overflow: 'hidden', position: 'relative',
  },
  barFill: { height: '100%', borderRadius: 5 },
  thresholdMark: {
    position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: Colors.warmGray400,
  },
  barLegend: { flexDirection: 'row', justifyContent: 'space-between' },
  barLabel: { fontSize: 10, color: '#35e27e', fontWeight: '600' },
  contributorRow: {
    flexDirection: 'row', alignItems: 'center', gap: Spacing.sm,
    paddingVertical: 4,
  },
  contributorDot: { width: 8, height: 8, borderRadius: 4 },
  contributorText: { flex: 1 },
  contributorLabel: { fontSize: FontSize.sm, fontWeight: '600', color: Colors.textPrimary },
  contributorDesc: { fontSize: FontSize.xs, color: Colors.textMuted },
  contributorBar: {
    width: 60, height: 4, borderRadius: 2, backgroundColor: Colors.warmGray100,
    overflow: 'hidden',
  },
  contributorBarFill: { height: '100%', borderRadius: 2 },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm, paddingVertical: 2 },
  infoText: { fontSize: FontSize.sm, color: Colors.textSecondary, flex: 1 },
  disclaimer: {
    flexDirection: 'row', gap: Spacing.sm, alignItems: 'flex-start',
    backgroundColor: Colors.warmGray100,
    borderRadius: Radius.md, padding: Spacing.md,
  },
  disclaimerText: {
    fontSize: FontSize.xs, color: Colors.textMuted, lineHeight: 18, flex: 1,
  },
  phqBanner: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: Colors.warmWhite, borderRadius: Radius.lg,
    padding: Spacing.md, borderWidth: 1, borderColor: Colors.softBlueMuted,
  },
  phqLeft: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md },
  phqTitle: { fontSize: FontSize.md, fontWeight: FontWeight.bold, color: Colors.textPrimary },
  phqSub: { fontSize: FontSize.xs, color: Colors.textSecondary },
});
