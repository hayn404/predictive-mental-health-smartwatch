import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  Image,
  useWindowDimensions,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, Radius } from '@/constants/theme';
import { useWellness } from '@/hooks/useWellness';
import { FocusGauge } from '@/components/ui/FocusGauge';
import { getFocusColor } from '@/services/ai/focusModel';

const TAB_BAR_HEIGHT = 82;

export default function FocusScreen() {
  const insets = useSafeAreaInsets();
  const { height: screenH } = useWindowDimensions();
  const { focus } = useWellness();

  // Static card height: screen minus every fixed element (header, tagline, features bar, gaps, padding)
  // header≈50  tagline≈60  featuresBar≈122  gaps(12×3)=36  paddingTop=insets.top+16  paddingBottom=insets.bottom+82
  const cardHeight = screenH - (insets.top + 16) - (insets.bottom + TAB_BAR_HEIGHT) - 50 - 60 - 122 - 36;

  const [lastTip, setLastTip]       = useState<string | null>(null);
  const [gaugeZoneH, setGaugeZoneH] = useState(0);

  useEffect(() => {
    if (focus.groqTips[0]) setLastTip(focus.groqTips[0]);
  }, [focus.groqTips]);

  const displayTip = focus.groqTips[0] ?? lastTip ?? (
    focus.focusScore >= 75
      ? 'Your signals look settled — a good moment to begin'
      : 'A few slow breaths before you start can bring your system into balance'
  );

  const hrEntry   = focus.elevatedFeatures.find(f => f.feature === 'hrMean');
  const hrvEntry  = focus.elevatedFeatures.find(f => f.feature === 'rmssd');
  const tempEntry = focus.elevatedFeatures.find(f => f.feature === 'tempMean');
  const balEntry  = focus.elevatedFeatures.find(f => f.feature === 'lfHfRatio');

  const sigColor = (e: typeof hrEntry) => e ? '#F59E0B' : '#35e27e';
  const accent   = getFocusColor(focus.focusLevel);

  // Gauge fills the measured zone minus a small breathing margin
  const gaugeSize = gaugeZoneH > 0 ? Math.min(270, Math.max(140, gaugeZoneH - 20)) : 180;

  return (
    <View style={[
      styles.root,
      { paddingTop: insets.top + 16, paddingBottom: insets.bottom + TAB_BAR_HEIGHT },
    ]}>

      {/* ── Header ── */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Image source={require('@/assets/images/seren-brain.png')} style={styles.appIcon} />
          <Text style={styles.title}>Focus</Text>
        </View>
        <View style={styles.badge}>
          <MaterialIcons name="verified-user" size={11} color={Colors.sageGreenDark} />
          <Text style={styles.badgeText}>ON-DEVICE</Text>
        </View>
      </View>

      {/* ── Tagline ── */}
      <Text style={styles.tagline}>
        Your anti-distraction companion — before and during exams or deep work.
        I read your body&apos;s live signals so you stay sharp throughout.
      </Text>

      {/* ── Readiness Card — fills all remaining space ── */}
      <View style={[styles.card, { borderColor: accent + '35', height: cardHeight }]}>

        <View style={styles.cardLabelRow}>
          <View style={[styles.dot, { backgroundColor: accent }]} />
          <Text style={styles.cardLabel}>COGNITIVE READINESS</Text>
        </View>

        <View
          style={styles.gaugeZone}
          onLayout={e => setGaugeZoneH(e.nativeEvent.layout.height)}
        >
          {gaugeZoneH > 0 && (
            <FocusGauge value={focus.focusScore} level={focus.focusLevel} size={gaugeSize} />
          )}
        </View>

        <View style={[styles.dividerH, { backgroundColor: accent + '28' }]} />

        <View style={styles.tipZone}>
          {focus.groqLoading ? (
            <ActivityIndicator size="small" color="#B0B7C3" />
          ) : (
            <View style={styles.tipRow}>
              <MaterialIcons name="auto-awesome" size={14} color={accent} style={{ marginTop: 3 }} />
              <Text style={styles.tipText}>{displayTip}</Text>
            </View>
          )}
        </View>

      </View>

      {/* ── Features Bar ── */}
      <View style={styles.featuresBar}>
        <View style={styles.featuresHeader}>
          <Text style={styles.featuresLabel}>LIVE SIGNALS</Text>
          <View style={styles.liveIndicator}>
            <View style={[styles.liveDot, { backgroundColor: accent }]} />
            <Text style={[styles.liveText, { color: accent }]}>active</Text>
          </View>
        </View>
        <View style={styles.featuresRow}>
          <FeatureItem icon="favorite"   label="Heart Rate" value={hrEntry?.value.toFixed(0)  ?? '—'} unit="bpm" color={sigColor(hrEntry)} />
          <View style={styles.featureDivider} />
          <FeatureItem icon="show-chart" label="HRV"        value={hrvEntry?.value.toFixed(0)  ?? '—'} unit="ms"  color={sigColor(hrvEntry)} />
          <View style={styles.featureDivider} />
          <FeatureItem icon="thermostat" label="Temp"       value={tempEntry?.value.toFixed(1) ?? '—'} unit="°C"  color={sigColor(tempEntry)} />
          <View style={styles.featureDivider} />
          <FeatureItem icon="device-hub" label="Balance"    value={balEntry?.value.toFixed(1)  ?? '—'} unit=""    color={sigColor(balEntry)} />
        </View>
      </View>

    </View>
  );
}

// ── Feature Item ─────────────────────────────────────────────

function FeatureItem({
  icon, label, value, unit, color,
}: { icon: string; label: string; value: string; unit: string; color: string }) {
  return (
    <View style={styles.featureItem}>
      <MaterialIcons name={icon as any} size={20} color={color} />
      <View style={styles.featureValues}>
        <Text style={[styles.featureValue, { color }]}>{value}</Text>
        {unit ? <Text style={[styles.featureUnit, { color }]}>{unit}</Text> : null}
      </View>
      <Text style={styles.featureLabel}>{label}</Text>
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────

const styles = StyleSheet.create({

  root: {
    flex: 1,
    paddingHorizontal: 20,
    gap: 12,
    backgroundColor: Colors.cream,
  },

  /* Header */
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  appIcon: { width: 34, height: 34, resizeMode: 'contain', borderRadius: Radius.sm },
  title: { fontSize: 24, fontWeight: '700', color: Colors.textPrimary, letterSpacing: -0.3 },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    borderWidth: 1,
    borderColor: Colors.sageGreenLight,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    backgroundColor: Colors.warmWhite,
  },
  badgeText: { fontSize: 10, color: Colors.sageGreenDark, fontWeight: '700', letterSpacing: 0.5 },

  /* Tagline */
  tagline: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
    fontWeight: '400',
  },

  /* Readiness card — height injected inline from cardHeight */
  card: {
    backgroundColor: Colors.warmWhite,
    borderRadius: 28,
    borderWidth: 1,
    paddingTop: 18,
    paddingBottom: 20,
    paddingHorizontal: 24,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  cardLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 8,
  },
  dot: { width: 7, height: 7, borderRadius: 4 },
  cardLabel: { fontSize: 10, fontWeight: '700', letterSpacing: 2, color: Colors.textSecondary },
  gaugeZone: {
    flex: 1,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dividerH: { width: '52%', height: 1, marginBottom: 2 },
  tipZone: {
    width: '100%',
    alignItems: 'center',
    minHeight: 44,
    justifyContent: 'center',
  },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 9, paddingHorizontal: 4 },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 21,
    fontWeight: '400',
  },

  /* Features bar */
  featuresBar: {
    backgroundColor: Colors.warmWhite,
    borderRadius: 24,
    paddingTop: 14,
    paddingBottom: 16,
    paddingHorizontal: 18,
    gap: 12,
  },
  featuresHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  featuresLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.8,
    color: Colors.textSecondary,
  },
  liveIndicator: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  liveDot: { width: 6, height: 6, borderRadius: 3 },
  liveText: { fontSize: 10, fontWeight: '600', letterSpacing: 0.3 },
  featuresRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
  },
  featureDivider: { width: 1, height: 40, backgroundColor: '#EFEFEF' },
  featureItem: { alignItems: 'center', gap: 4, flex: 1 },
  featureValues: { flexDirection: 'row', alignItems: 'baseline', gap: 2 },
  featureValue: { fontSize: 18, fontWeight: '700', lineHeight: 22 },
  featureUnit: { fontSize: 9, fontWeight: '600', opacity: 0.75 },
  featureLabel: { fontSize: 10, color: Colors.textSecondary, fontWeight: '500' },
});
