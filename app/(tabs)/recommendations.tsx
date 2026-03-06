import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Linking,
  Image,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Colors, FontSize, FontWeight, Spacing, Radius, Shadow } from '@/constants/theme';
import { useHealthData } from '@/hooks/useHealthData';
import { GlassCard } from '@/components/ui/GlassCard';
import { Recommendation } from '@/services/mockData';
import { useAlert } from '@/template';

const CATEGORY_LABELS: Record<string, string> = {
  all: 'All',
  breathing: 'Breathing',
  physical: 'Physical',
  journaling: 'Journaling',
  meditation: 'Meditation',
};

const CATEGORY_ICONS: Record<string, string> = {
  breathing: 'air',
  physical: 'directions-run',
  journaling: 'edit-note',
  meditation: 'self-improvement',
};

const CATEGORY_COLORS: Record<string, string> = {
  breathing: Colors.softBlue,
  physical: Colors.sageGreen,
  journaling: Colors.violet,
  meditation: Colors.warning,
};

export default function RecommendationsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { showAlert } = useAlert();

  const handleGetHelp = () => {
    showAlert(
      'Crisis Support',
      'If you are in immediate danger, please call emergency services.\n\n• National Suicide Prevention Lifeline: 988\n• Crisis Text Line: Text HOME to 741741\n• Emergency: 911',
      [
        { text: 'Call 988', onPress: () => Linking.openURL('tel:988') },
        { text: 'Close', style: 'cancel' },
      ]
    );
  };

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
              <Image source={require('@/assets/images/logo.png')} style={styles.appIcon} />
            </View>
            <Text style={styles.title}>For You</Text>
          </View>
          <View style={styles.localBadge}>
            <MaterialIcons name="verified-user" size={14} color={Colors.sageGreenDark} />
            <Text style={styles.localText}>LOCAL</Text>
          </View>
        </View>

        {/* AI SUGGESTION */}
        <View style={styles.sectionHeaderRow}>
          <View style={styles.sectionHeaderLeft}>
            <MaterialIcons name="auto-awesome" size={16} color={'#A78BFA'} />
            <Text style={styles.sectionLabelHeader}>AI SUGGESTION</Text>
          </View>
          <View style={styles.newInsightPill}>
            <Text style={styles.newInsightText}>New Insight</Text>
          </View>
        </View>

        <View style={styles.insightCard}>
          <View style={styles.insightContentRow}>
            <View style={styles.insightIconWrapper}>
              <MaterialIcons name="show-chart" size={24} color={Colors.warmWhite} />
            </View>
            <View style={styles.insightTexts}>
              <Text style={styles.insightQuote}>"Based on your high stress at 3 PM daily, we recommend a short walk then."</Text>
              <Text style={styles.insightSub}>Matches your typical HRV dip window.</Text>
            </View>
          </View>
          <View style={styles.insightActions}>
            <TouchableOpacity style={styles.scheduleBtn}>
              <Text style={styles.scheduleBtnText}>Schedule Walk</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.goBtn}>
              <MaterialIcons name="arrow-forward" size={18} color={'#8B5CF6'} />
            </TouchableOpacity>
          </View>
        </View>

        {/* EXPLORE TOOLS */}
        <View style={[styles.sectionHeaderRow, { marginTop: Spacing.xl }]}>
          <Text style={styles.sectionLabelHeader}>EXPLORE TOOLS</Text>
          <TouchableOpacity>
            <Text style={styles.viewAllText}>View All {'>'}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.toolsGrid}>
          <View style={styles.toolCard}>
            <MaterialIcons name="air" size={20} color={Colors.textPrimary} style={styles.toolIcon} />
            <Text style={styles.toolTitleText}>Breathing</Text>
            <Text style={styles.toolDescText}>Calm your nervous system with rhythmic</Text>
            <View style={styles.toolFooter}>
              <View style={styles.toolDurationRow}>
                <MaterialIcons name="schedule" size={12} color={Colors.textMuted} />
                <Text style={styles.toolDurationText}>3-5m</Text>
              </View>
              <View style={styles.toolGoBtn}>
                <MaterialIcons name="chevron-right" size={16} color={Colors.textPrimary} />
              </View>
            </View>
          </View>

          <View style={styles.toolCard}>
            <MaterialIcons name="edit" size={20} color={Colors.textPrimary} style={styles.toolIcon} />
            <Text style={styles.toolTitleText}>Journaling</Text>
            <Text style={styles.toolDescText}>Externalize your thoughts and identify</Text>
            <View style={styles.toolFooter}>
              <View style={styles.toolDurationRow}>
                <MaterialIcons name="schedule" size={12} color={Colors.textMuted} />
                <Text style={styles.toolDurationText}>10m</Text>
              </View>
              <View style={styles.toolGoBtn}>
                <MaterialIcons name="chevron-right" size={16} color={Colors.textPrimary} />
              </View>
            </View>
          </View>

          <View style={styles.toolCard}>
            <MaterialIcons name="show-chart" size={20} color={Colors.sageGreen} style={styles.toolIcon} />
            <Text style={styles.toolTitleText}>Physical</Text>
            <Text style={styles.toolDescText}>Burn off cortisol with light cardiovascular</Text>
            <View style={styles.toolFooter}>
              <View style={styles.toolDurationRow}>
                <MaterialIcons name="schedule" size={12} color={Colors.textMuted} />
                <Text style={styles.toolDurationText}>15m</Text>
              </View>
              <View style={styles.toolGoBtn}>
                <MaterialIcons name="chevron-right" size={16} color={Colors.textPrimary} />
              </View>
            </View>
          </View>

          <View style={styles.toolCard}>
            <MaterialIcons name="favorite-border" size={20} color={'#A78BFA'} style={styles.toolIcon} />
            <Text style={styles.toolTitleText}>Meditation</Text>
            <Text style={styles.toolDescText}>Practice presence with guided mindfulness</Text>
            <View style={styles.toolFooter}>
              <View style={styles.toolDurationRow}>
                <MaterialIcons name="schedule" size={12} color={Colors.textMuted} />
                <Text style={styles.toolDurationText}>5-20m</Text>
              </View>
              <View style={styles.toolGoBtn}>
                <MaterialIcons name="chevron-right" size={16} color={Colors.textPrimary} />
              </View>
            </View>
          </View>
        </View>

        {/* MORNING RITUALS */}
        <Text style={[styles.sectionLabelHeader, { marginTop: Spacing.xl, marginBottom: Spacing.md }]}>MORNING RITUALS</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.ritualsScroll}>
          <View style={styles.ritualItem}>
            <View style={styles.ritualImageContainer}>
              <Image source={{ uri: 'https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?q=80&w=400' }} style={styles.ritualImage} />
              <View style={styles.ritualOverlay}>
                <Text style={styles.ritualOverlayText}>Boost Circadian Rhythm</Text>
              </View>
            </View>
            <Text style={styles.ritualTitle}>Sunlight Exposure</Text>
          </View>

          <View style={styles.ritualItem}>
            <View style={styles.ritualImageContainer}>
              <Image source={{ uri: 'https://images.unsplash.com/photo-1550596334-7bb40a71b6bc?q=80&w=400' }} style={styles.ritualImage} />
              <View style={styles.ritualOverlay}>
                <Text style={styles.ritualOverlayText}>Reset Nervous System</Text>
              </View>
            </View>
            <Text style={styles.ritualTitle}>Cold Splash</Text>
          </View>

          <View style={styles.ritualItem}>
            <View style={styles.ritualImageContainer}>
              <Image source={{ uri: 'https://images.unsplash.com/photo-1506126613408-eca07ce68773?q=80&w=400' }} style={styles.ritualImage} />
              <View style={styles.ritualOverlay}>
                <Text style={styles.ritualOverlayText}>Positive Focus</Text>
              </View>
            </View>
            <Text style={styles.ritualTitle}>Gratitude</Text>
          </View>
        </ScrollView>

        <View style={styles.divider} />

        {/* Support Card */}
        <View style={styles.supportCard}>
          <MaterialIcons name="help-outline" size={20} color={Colors.textSecondary} style={{ marginTop: 2, marginLeft: 4 }} />
          <View style={styles.supportTextCol}>
            <Text style={styles.supportTitle}>Need immediate support?</Text>
            <Text style={styles.supportDesc}>Resources for when you're overwhelmed.</Text>
          </View>
          <TouchableOpacity style={styles.supportBtn} onPress={handleGetHelp}>
            <Text style={styles.supportBtnText}>Get Help Now</Text>
          </TouchableOpacity>
        </View>

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F3F4F6',
  },
  content: {
    paddingHorizontal: Spacing.xl,
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
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: -Spacing.sm,
  },
  sectionHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  sectionLabelHeader: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.textSecondary,
    letterSpacing: 0.5,
  },
  newInsightPill: {
    backgroundColor: '#F5F3FF',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  newInsightText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#8B5CF6',
  },
  viewAllText: {
    fontSize: FontSize.sm,
    fontWeight: '600',
    color: Colors.sageGreen,
  },
  insightCard: {
    backgroundColor: '#F5F3FF',
    borderRadius: Radius.xl,
    padding: Spacing.lg,
    marginTop: Spacing.sm,
  },
  insightContentRow: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginBottom: Spacing.md,
  },
  insightIconWrapper: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#8B5CF6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  insightTexts: {
    flex: 1,
  },
  insightQuote: {
    fontSize: FontSize.md,
    color: '#4C1D95',
    fontWeight: '600',
    lineHeight: 22,
    marginBottom: 4,
  },
  insightSub: {
    fontSize: FontSize.xs,
    color: '#7C3AED',
  },
  insightActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  scheduleBtn: {
    flex: 1,
    backgroundColor: '#8B5CF6',
    borderRadius: Radius.full,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scheduleBtnText: {
    color: Colors.warmWhite,
    fontSize: FontSize.sm,
    fontWeight: 'bold',
  },
  goBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.warmWhite,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toolsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: Spacing.md,
  },
  toolCard: {
    width: '47.5%',
    backgroundColor: Colors.warmWhite,
    borderRadius: Radius.xl,
    padding: Spacing.md,
    paddingVertical: Spacing.lg,
    borderWidth: 1,
    borderColor: '#F3F4F6',
    shadowColor: Colors.textPrimary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.03,
    shadowRadius: 10,
    elevation: 2,
  },
  toolIcon: {
    marginBottom: Spacing.md,
  },
  toolTitleText: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  toolDescText: {
    fontSize: 11,
    color: Colors.textSecondary,
    lineHeight: 16,
    marginBottom: Spacing.lg,
    height: 32,
  },
  toolFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 'auto',
  },
  toolDurationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  toolDurationText: {
    fontSize: 10,
    fontWeight: '500',
    color: Colors.textMuted,
  },
  toolGoBtn: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#F3F4F6',
    alignItems: 'center',
    justifyContent: 'center',
  },
  ritualsScroll: {
    gap: Spacing.md,
    paddingRight: Spacing.xl,
  },
  ritualItem: {
    width: 140,
    gap: Spacing.xs,
  },
  ritualImageContainer: {
    width: '100%',
    height: 160,
    borderRadius: Radius.lg,
    overflow: 'hidden',
    position: 'relative',
  },
  ritualImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  ritualOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: Spacing.sm,
    paddingTop: 20,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  ritualOverlayText: {
    color: Colors.warmWhite,
    fontSize: 10,
    fontWeight: 'bold',
  },
  ritualTitle: {
    fontSize: FontSize.sm,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  divider: {
    height: 1,
    backgroundColor: Colors.warmGray200,
    marginVertical: Spacing.md,
  },
  supportCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FCFBFC',
    borderRadius: Radius.lg,
    padding: Spacing.md,
    gap: Spacing.md,
  },
  supportTextCol: {
    flex: 1,
    gap: 2,
  },
  supportTitle: {
    fontSize: FontSize.sm,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  supportDesc: {
    fontSize: 10,
    color: Colors.textSecondary,
    lineHeight: 14,
  },
  supportBtn: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: Radius.full,
    borderWidth: 1,
    borderColor: '#FCA5A5',
  },
  supportBtnText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#EF4444',
  },
});
