import React, { useState } from 'react';
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
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { useHealthData } from '@/hooks/useHealthData';
import { GlassCard } from '@/components/ui/GlassCard';
import { TrendChart, BarChart, SleepHeatmap } from '@/components/ui/TrendChart';

const TABS = ['Daily', 'Weekly', 'Monthly'] as const;
type TabType = typeof TABS[number];

export default function InsightsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { weeklyData, monthlySleep, hrvTrend, weeklyLocationDiversity, weeklySunlight } = useHealthData();
  const [activeTab, setActiveTab] = useState<TabType>('Daily');

  const stressChartData = weeklyData.map(d => ({ label: d.date, value: d.stress }));
  const sleepChartData = weeklyData.map(d => ({ label: d.date, value: d.sleep }));
  const hrvChartData = hrvTrend.map(d => ({ label: d.time, value: d.value }));
  const weeklyBarData = weeklyData.map(d => ({
    label: d.date,
    value: d.stress,
    color: d.stress > 60 ? Colors.error : d.stress > 40 ? Colors.warning : Colors.sageGreen,
  }));

  // Create a mock dataset for the purple Stress Level line based on the HRV timeline
  const mockStressCurveData = hrvTrend.map((d, i) => ({
    label: d.time,
    value: Math.max(15, Math.min(100, (d.value * 0.8) + (i % 2 === 0 ? 25 : -15)))
  }));

  return (
    <View style={styles.container}>
      <ScrollView
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
            <Text style={styles.title}>Historical Trends</Text>
          </View>
          <View style={styles.localBadge}>
            <MaterialIcons name="verified-user" size={14} color={Colors.sageGreenDark} />
            <Text style={styles.localText}>LOCAL</Text>
          </View>
        </View>

        {/* Filters */}
        <View style={styles.tabsContainer}>
          {TABS.map(tab => (
            <TouchableOpacity
              key={tab}
              style={[styles.filterChip, activeTab === tab && styles.filterChipActive]}
              onPress={() => setActiveTab(tab)}
            >
              <Text style={[styles.filterChipText, activeTab === tab && styles.filterChipTextActive]}>{tab}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Bio-Correlation */}
        <View style={styles.chartCard}>
          <View style={styles.chartHeaderRow}>
            <View style={styles.chartTitleContainer}>
              <MaterialIcons name="show-chart" size={20} color={Colors.sageGreen} />
              <View>
                <Text style={styles.chartTitle}>Bio-Correlation</Text>
                <Text style={styles.chartSubTitle}>HR (BPM) vs. Stress Intensity</Text>
              </View>
            </View>
            <View style={styles.chartPeakContainer}>
              <Text style={styles.peakStressText}>Peak Stress: 65%</Text>
              <Text style={styles.peakTimeText}>at 02:00 PM</Text>
            </View>
          </View>

          <View style={styles.chartPlaceholder}>
            {/* Dual-line chart overlaying two TrendCharts */}
            <View style={styles.chartLineMockup}>
              <TrendChart
                data={hrvChartData}
                color={Colors.sageGreen}
                height={180}
                maxValue={100}
                showDots={false}
              />
              <View style={StyleSheet.absoluteFill}>
                <TrendChart
                  data={mockStressCurveData}
                  color={'#A78BFA'}
                  height={180}
                  maxValue={100}
                  showDots={true}
                  showArea={false}
                />
              </View>
            </View>
            <View style={styles.axisHorizontalLabels}>
              <Text style={styles.axisLabel}>08:00</Text>
              <Text style={styles.axisLabel}>10:00</Text>
              <Text style={styles.axisLabel}>12:00</Text>
              <Text style={styles.axisLabel}>14:00</Text>
              <Text style={styles.axisLabel}>16:00</Text>
              <Text style={styles.axisLabel}>18:00</Text>
              <Text style={styles.axisLabel}>20:00</Text>
            </View>
          </View>

          <View style={styles.chartLegend}>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: Colors.sageGreen }]} />
              <Text style={styles.legendText}>Heart Rate</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: '#A78BFA' }]} />
              <Text style={styles.legendText}>Stress Level</Text>
            </View>
          </View>
        </View>

        {/* Sleep Consistency */}
        <View style={styles.chartCard}>
          <View style={styles.chartHeaderRow}>
            <View style={styles.chartTitleContainer}>
              <MaterialIcons name="bedtime" size={20} color={Colors.softBlue} />
              <View>
                <Text style={styles.chartTitle}>Sleep Consistency</Text>
                <Text style={styles.chartSubTitle}>Last 31 days consistency</Text>
              </View>
            </View>
            <View style={styles.sleepBadge}>
              <Text style={styles.sleepBadgeText}>84% Avg.</Text>
            </View>
          </View>

          <View style={styles.sleepGridMockup}>
            <SleepHeatmap data={monthlySleep} />
          </View>

          <View style={styles.sleepLegendRow}>
            <Text style={styles.sleepLegendLabel}>LOW QUALITY</Text>
            <View style={styles.sleepLegendScale}>
              <View style={[styles.sleepLegendDot, { backgroundColor: '#F1F5F9' }]} />
              <View style={[styles.sleepLegendDot, { backgroundColor: '#DBEAFE' }]} />
              <View style={[styles.sleepLegendDot, { backgroundColor: '#93C5FD' }]} />
              <View style={[styles.sleepLegendDot, { backgroundColor: '#60A5FA' }]} />
            </View>
            <Text style={styles.sleepLegendLabel}>HIGH QUALITY</Text>
          </View>
        </View>

        {/* Weekly Sunlight Exposure */}
        <View style={styles.chartCard}>
          <View style={styles.chartHeaderRow}>
            <View style={styles.chartTitleContainer}>
              <MaterialIcons name="wb-sunny" size={20} color={Colors.warning} />
              <View>
                <Text style={styles.chartTitle}>Sunlight Exposure</Text>
                <Text style={styles.chartSubTitle}>Minutes outdoors per day</Text>
              </View>
            </View>
            <View style={[styles.sleepBadge, { backgroundColor: '#FEF3C7' }]}>
              <Text style={[styles.sleepBadgeText, { color: '#92400E' }]}>Goal: 30m</Text>
            </View>
          </View>

          {weeklySunlight.length > 0 ? (
            <BarChart
              color={Colors.warning}
              data={weeklySunlight.map(d => ({
                label: d.date,
                value: d.value,
                color: d.value >= 30 ? Colors.sageGreen : d.value >= 15 ? Colors.warning : Colors.error,
              }))}
            />
          ) : (
            <View style={styles.emptyChartPlaceholder}>
              <MaterialIcons name="wb-sunny" size={32} color={Colors.textMuted} />
              <Text style={styles.emptyChartText}>Sunlight data will appear here</Text>
            </View>
          )}
        </View>

        {/* Weekly Location Diversity */}
        <View style={styles.chartCard}>
          <View style={styles.chartHeaderRow}>
            <View style={styles.chartTitleContainer}>
              <MaterialIcons name="place" size={20} color={Colors.violet} />
              <View>
                <Text style={styles.chartTitle}>Location Diversity</Text>
                <Text style={styles.chartSubTitle}>Variety of places visited</Text>
              </View>
            </View>
          </View>

          {weeklyLocationDiversity.length > 0 ? (
            <BarChart
              color={Colors.violet}
              data={weeklyLocationDiversity.map(d => ({
                label: d.date,
                value: d.value,
                color: d.value >= 50 ? Colors.sageGreen : d.value >= 20 ? Colors.warning : Colors.error,
              }))}
            />
          ) : (
            <View style={styles.emptyChartPlaceholder}>
              <MaterialIcons name="place" size={32} color={Colors.textMuted} />
              <Text style={styles.emptyChartText}>Location data will appear here</Text>
            </View>
          )}

          <View style={styles.insightTipRow}>
            <MaterialIcons name="info-outline" size={14} color={Colors.violet} />
            <Text style={styles.insightTipText}>
              Research shows that visiting diverse locations is linked to better mood and lower depression risk.
            </Text>
          </View>
        </View>

        {/* What is HRV? */}
        <View style={styles.hrvCard}>
          <View style={styles.hrvHeader}>
            <View style={styles.hrvIconBg}>
              <MaterialIcons name="psychology" size={20} color={'#A78BFA'} />
            </View>
            <Text style={styles.hrvTitle}>What is HRV?</Text>
          </View>
          <Text style={styles.hrvText}>
            Heart Rate Variability (HRV) measures the specific time changes between each heartbeat. It's a gold standard for mapping your nervous system's resilience.
          </Text>
          <View style={styles.hrvStatesRow}>
            <View style={[styles.hrvStateBox, { marginRight: Spacing.sm }]}>
              <Text style={[styles.hrvStateTitle, { color: Colors.sageGreen }]}>HIGH HRV</Text>
              <Text style={styles.hrvStateDesc}>Better recovery & lower stress levels.</Text>
            </View>
            <View style={styles.hrvStateBox}>
              <Text style={[styles.hrvStateTitle, { color: '#C084FC' }]}>LOW HRV</Text>
              <Text style={styles.hrvStateDesc}>Sign of fatigue or physiological stress.</Text>
            </View>
          </View>
        </View>

        <TouchableOpacity style={styles.exportBtn}>
          <MaterialIcons name="info-outline" size={16} color={Colors.textSecondary} />
          <Text style={styles.exportText}>Export Monthly Privacy Report</Text>
          <MaterialIcons name="chevron-right" size={16} color={Colors.textSecondary} style={{ marginLeft: 'auto' }} />
        </TouchableOpacity>

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.cream,
  },
  content: {
    paddingHorizontal: Spacing.xl,
    gap: Spacing.xl,
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
    gap: Spacing.md,
  },
  appIconContainer: {
    width: 32,
    height: 32,
    borderRadius: Radius.sm,
    overflow: 'hidden',
    backgroundColor: Colors.warmWhite,
  },
  appIcon: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  title: {
    fontSize: FontSize.lg,
    fontWeight: 'bold',
    color: Colors.textPrimary,
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
  tabsContainer: {
    flexDirection: 'row',
    backgroundColor: '#F3F4F6',
    borderRadius: Radius.full,
    padding: 4,
    borderWidth: 1,
    borderColor: '#E2E8F0',
  },
  filterChip: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: Radius.full,
  },
  filterChipActive: {
    backgroundColor: Colors.warmWhite,
  },
  filterChipText: {
    fontSize: FontSize.sm,
    fontWeight: '500',
    color: Colors.textSecondary,
  },
  filterChipTextActive: {
    color: Colors.textPrimary,
    fontWeight: '600',
  },
  chartCard: {
    backgroundColor: Colors.warmWhite,
    borderRadius: Radius.xl,
    padding: Spacing.lg,
  },
  chartHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.lg,
  },
  chartTitleContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  chartTitle: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  chartSubTitle: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
    marginTop: 2,
  },
  chartPeakContainer: {
    alignItems: 'flex-end',
  },
  peakStressText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#A78BFA',
  },
  peakTimeText: {
    fontSize: 10,
    color: Colors.textMuted,
  },
  sleepBadge: {
    backgroundColor: '#DBEAFE',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  sleepBadgeText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: Colors.softBlueDark,
  },
  chartPlaceholder: {
    marginVertical: Spacing.md,
  },
  chartLineMockup: {
    height: 180,
    borderBottomWidth: 1,
    borderBottomColor: Colors.warmGray100,
    marginBottom: Spacing.sm,
  },
  axisHorizontalLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  axisLabel: {
    fontSize: 9,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  chartLegend: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xl,
    marginTop: Spacing.md,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  legendText: {
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
    fontWeight: '500',
  },
  sleepGridMockup: {
    marginVertical: Spacing.md,
  },
  sleepLegendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: Spacing.md,
  },
  sleepLegendLabel: {
    fontSize: 9,
    fontWeight: 'bold',
    color: Colors.textSecondary,
    letterSpacing: 0.5,
  },
  sleepLegendScale: {
    flexDirection: 'row',
    gap: 4,
  },
  sleepLegendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  hrvCard: {
    backgroundColor: '#F5F3FF',
    borderRadius: Radius.xl,
    padding: Spacing.xl,
  },
  hrvHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.md,
  },
  hrvIconBg: {
    width: 36,
    height: 36,
    borderRadius: Radius.full,
    backgroundColor: '#EDE9FE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  hrvTitle: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: '#8B5CF6',
  },
  hrvText: {
    fontSize: FontSize.sm,
    color: '#6D28D9',
    lineHeight: 20,
    marginBottom: Spacing.lg,
  },
  hrvStatesRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  hrvStateBox: {
    flex: 1,
    backgroundColor: Colors.warmWhite,
    padding: Spacing.md,
    borderRadius: Radius.lg,
  },
  hrvStateTitle: {
    fontSize: 10,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  hrvStateDesc: {
    fontSize: 11,
    color: Colors.textSecondary,
    lineHeight: 16,
  },
  exportBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.warmWhite,
    borderWidth: 1,
    borderColor: Colors.warmGray200,
    borderStyle: 'dashed',
    padding: Spacing.md,
    borderRadius: Radius.full,
    gap: Spacing.sm,
  },
  exportText: {
    fontSize: FontSize.sm,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  emptyChartPlaceholder: {
    height: 120,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
  },
  emptyChartText: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
  },
  insightTipRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.sm,
    marginTop: Spacing.md,
    paddingTop: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: '#F3F4F6',
  },
  insightTipText: {
    flex: 1,
    fontSize: FontSize.xs,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
});
