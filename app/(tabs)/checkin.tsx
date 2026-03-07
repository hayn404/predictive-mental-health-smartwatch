import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
  KeyboardAvoidingView,
  Platform,
  Image,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Colors, FontSize, FontWeight, Spacing, Radius, Shadow } from '@/constants/theme';
import { useCheckin } from '@/hooks/useHealthData';
import { GlassCard } from '@/components/ui/GlassCard';
import { getCheckinHistory } from '@/services/mockData';

export default function CheckinScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isRecording, transcript, isAnalyzing, result, waveAmplitudes, startRecording, stopAndAnalyze, reset } = useCheckin();
  const history = getCheckinHistory();
  const micPulse = useRef(new Animated.Value(1)).current;
  const waveAnims = useRef(waveAmplitudes.map(() => new Animated.Value(4))).current;

  useEffect(() => {
    if (isRecording) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(micPulse, { toValue: 1.12, duration: 700, useNativeDriver: true }),
          Animated.timing(micPulse, { toValue: 1, duration: 700, useNativeDriver: true }),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    }
  }, [isRecording]);

  useEffect(() => {
    waveAmplitudes.forEach((amp, i) => {
      Animated.timing(waveAnims[i], {
        toValue: amp,
        duration: 80,
        useNativeDriver: false,
      }).start();
    });
  }, [waveAmplitudes]);

  const sentimentColors: Record<string, string> = {
    positive: Colors.sageGreen,
    neutral: Colors.softBlue,
    concerned: Colors.warning,
  };

  const formatTimeAgo = (date: Date) => {
    const mins = Math.round((Date.now() - date.getTime()) / 60000);
    if (mins < 60) return `${mins}m ago`;
    return `${Math.round(mins / 60)}h ago`;
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
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
            <TouchableOpacity onPress={() => router.push('/(tabs)')} style={{ marginRight: 4 }}>
              <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
            </TouchableOpacity>
            <View style={styles.appIconContainer}>
              <Image source={require('@/assets/images/logo.png')} style={styles.appIcon} />
            </View>
            <Text style={styles.title}>Check-in</Text>
          </View>
          <View style={styles.localBadge}>
            <MaterialIcons name="verified-user" size={14} color={Colors.sageGreenDark} />
            <Text style={styles.localText}>LOCAL</Text>
          </View>
        </View>

        {!result && !isAnalyzing && (
          <View style={styles.voiceInteractionArea}>
            {/* Mic Box */}
            <View style={styles.micGlowContainer}>
              <TouchableOpacity
                style={[
                  styles.micButton,
                  isRecording && { backgroundColor: Colors.errorMuted }
                ]}
                onPress={isRecording ? stopAndAnalyze : startRecording}
                activeOpacity={0.85}
              >
                <MaterialIcons
                  name={isRecording ? "stop" : "mic"}
                  size={42}
                  color={isRecording ? Colors.error : Colors.sageGreen}
                />
              </TouchableOpacity>
            </View>
            <Text style={{ textAlign: 'center', marginTop: 16, color: Colors.textSecondary, fontSize: FontSize.sm, fontWeight: '600', letterSpacing: 0.5 }}>
              {isRecording ? "Tap square to stop & analyze" : "Tap microphone to begin"}
            </Text>

            {/* Waveform */}
            <View style={styles.waveform}>
              {waveAmplitudes.slice(0, 15).map((amp, i) => (
                <Animated.View
                  key={i}
                  style={[
                    styles.waveBar,
                    {
                      height: waveAnims[i],
                      backgroundColor: '#C4B5FD',
                      opacity: isRecording ? 1 : 0.5,
                    },
                  ]}
                />
              ))}
            </View>
            <Text style={styles.recordingActiveText}>• VOICE RECORDING ACTIVE</Text>

            {/* Transcript Card */}
            <View style={styles.transcriptCard}>
              <View style={styles.transcriptHeader}>
                <View style={styles.liveIndicator}>
                  <View style={[styles.liveIndicatorDot, { backgroundColor: isRecording ? '#A78BFA' : Colors.warmGray400 }]} />
                  <Text style={[styles.liveIndicatorText, { color: isRecording ? '#A78BFA' : Colors.textMuted }]}>CAPTURING TONE & TEXT</Text>
                </View>
                <View style={styles.privacyProtectedBadge}>
                  <Text style={styles.privacyProtectedText}>Privacy Protected</Text>
                </View>
              </View>
              <Text style={styles.transcriptContent}>
                {transcript || (!isRecording ? "How are you feeling today? I'm listening." : '')}
                {isRecording && <Text style={{ color: '#C4B5FD', fontSize: FontSize.lg }}> |</Text>}
              </Text>
            </View>

            {/* Secondary Controls */}
            <View style={styles.secondaryControls}>
              <TouchableOpacity style={styles.controlItem}>
                <MaterialIcons name="mic-off" size={20} color={Colors.textMuted} />
                <Text style={styles.controlLabel}>MUTE</Text>
              </TouchableOpacity>
              <View style={styles.controlDivider} />
              <TouchableOpacity style={styles.controlItem}>
                <MaterialIcons name="info-outline" size={20} color={Colors.textMuted} />
                <Text style={styles.controlLabel}>TIPS</Text>
              </TouchableOpacity>
            </View>

            {/* Footer */}
            <View style={styles.bottomFooter}>
              <View style={styles.e2eBadge}>
                <View style={[styles.insightDot, { backgroundColor: Colors.sageGreen }]} />
                <Text style={styles.e2eText}>END-TO-END ON-DEVICE NEURAL PROCESSING</Text>
              </View>
            </View>
          </View>
        )}

        {/* Analyzing */}
        {isAnalyzing && (
          <View style={styles.analyzingContainer}>
            <Animated.View style={styles.analyzingIcon}>
              <MaterialIcons name="psychology" size={44} color={Colors.violet} />
            </Animated.View>
            <Text style={styles.analyzingText}>Analyzing your check-in...</Text>
            <Text style={styles.analyzingSubtext}>Understanding context, tone & patterns</Text>
          </View>
        )}

        {/* Result */}
        {result && (
          <View style={styles.resultSection}>
            <GlassCard
              variant={result?.sentiment === 'positive' ? 'sage' : result?.sentiment === 'concerned' ? 'default' : 'blue'}
              style={styles.resultCard}
            >
              <View style={styles.resultHeader}>
                <Text style={styles.resultTitle}>Check-in Complete</Text>
                <View style={[styles.sentimentBadge, { backgroundColor: (result?.sentiment ? sentimentColors[result.sentiment] : Colors.sageGreen) + '20' }]}>
                  <Text style={[styles.sentimentText, { color: result?.sentiment ? sentimentColors[result.sentiment] : Colors.sageGreen }]}>
                    {result?.sentiment ? result.sentiment.charAt(0).toUpperCase() + result.sentiment.slice(1) : ''}
                  </Text>
                </View>
              </View>
              <Text style={styles.resultSubtitle}>Key Insights</Text>
              {result?.insights?.map((insight: string, i: number) => (
                <View key={i} style={styles.insightItem}>
                  <View style={[styles.insightDot, { backgroundColor: Colors.sageGreen }]} />
                  <Text style={styles.insightText}>{insight}</Text>
                </View>
              ))}
            </GlassCard>

            <TouchableOpacity style={styles.newCheckinBtn} onPress={reset}>
              <MaterialIcons name="refresh" size={18} color={Colors.violet} />
              <Text style={styles.newCheckinText}>New Check-in</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView >
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.cream,
  },
  content: {
    paddingHorizontal: Spacing.lg,
    gap: Spacing.lg,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
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
    fontWeight: FontWeight.bold,
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
  },
  localText: {
    fontSize: FontSize.xs,
    color: Colors.sageGreenDark,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  voiceInteractionArea: {
    alignItems: 'center',
    paddingTop: Spacing.xl,
    gap: Spacing.md,
  },
  micGlowContainer: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#F5F3FF', // Very subtle purple glow
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.lg,
  },
  micButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.warmWhite,
    alignItems: 'center',
    justifyContent: 'center',
  },
  waveform: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    height: 60,
  },
  waveBar: {
    width: 6,
    borderRadius: 3,
    minHeight: 8,
  },
  recordingActiveText: {
    fontSize: 10,
    color: '#A78BFA',
    fontWeight: 'bold',
    letterSpacing: 1,
    marginTop: Spacing.xs,
    textTransform: 'uppercase',
  },
  transcriptCard: {
    width: '100%',
    backgroundColor: Colors.warmWhite,
    borderRadius: Radius.xl,
    padding: Spacing.xl,
    marginTop: Spacing.md,
    borderWidth: 1.5,
    borderColor: '#F5F3FF',
    minHeight: 180,
  },
  transcriptHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  liveIndicatorDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  liveIndicatorText: {
    fontSize: 10,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },
  privacyProtectedBadge: {
    backgroundColor: '#DCFCE7', // Light green
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  privacyProtectedText: {
    color: '#166534',
    fontSize: 9,
    fontWeight: 'bold',
  },
  transcriptContent: {
    fontSize: FontSize.md,
    lineHeight: 24,
    color: Colors.textPrimary,
    fontWeight: '500',
  },
  endButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#8B5CF6',
    width: '100%',
    paddingVertical: 18,
    borderRadius: Radius.xl,
    marginTop: Spacing.xl,
  },
  endButtonText: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.warmWhite,
  },
  secondaryControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.xl,
    marginTop: Spacing.xl,
  },
  controlItem: {
    alignItems: 'center',
    gap: 4,
  },
  controlDivider: {
    width: 1,
    height: 30,
    backgroundColor: Colors.warmGray200,
    marginHorizontal: Spacing.md,
  },
  controlLabel: {
    fontSize: 9,
    color: Colors.textMuted,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },
  bottomFooter: {
    alignItems: 'center',
    marginTop: Spacing.xxl + 10,
    backgroundColor: Colors.warmWhite,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: Radius.full,
    borderWidth: 1,
    borderColor: '#F3F4F6',
    alignSelf: 'center',
  },
  e2eBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  e2eText: {
    fontSize: 8,
    color: Colors.textMuted,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },
  analyzingContainer: {
    alignItems: 'center',
    gap: Spacing.sm,
    marginTop: Spacing.xl,
  },
  analyzingIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: Colors.violetMuted,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.sm,
  },
  analyzingText: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  analyzingSubtext: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
  },
  resultSection: {
    gap: Spacing.md,
  },
  resultCard: {
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  resultTitle: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  sentimentBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  sentimentText: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    textTransform: 'capitalize',
  },
  resultSubtitle: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  insightItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  insightDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  insightText: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  newCheckinBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: Spacing.md,
    borderRadius: Radius.full,
    borderWidth: 1.5,
    borderColor: Colors.violet,
  },
  newCheckinText: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.semibold,
    color: Colors.violet,
  },
  sectionLabel: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  historyCard: {
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sentimentDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  historyTime: {
    flex: 1,
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  historyTranscript: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  historyInsights: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  insightChip: {
    backgroundColor: Colors.violetMuted,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  insightChipText: {
    fontSize: FontSize.xs,
    color: Colors.violetDark,
    fontWeight: FontWeight.medium,
  },
});
