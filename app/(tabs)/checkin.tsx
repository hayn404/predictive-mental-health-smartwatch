import React, { useEffect, useRef, useState } from 'react';
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
  TextInput,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Colors, FontSize, FontWeight, Spacing, Radius, Shadow } from '@/constants/theme';
import { useCheckin, useHealthData } from '@/hooks/useHealthData';
import { GlassCard } from '@/components/ui/GlassCard';

export default function CheckinScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isRecording, isTranscribing, transcript, isAnalyzing, result, waveAmplitudes, recordingDuration, startRecording, stopAndAnalyze, submitText, reset } = useCheckin();
  const { checkinHistory: history } = useHealthData();
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('text');
  const [textInput, setTextInput] = useState('');
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

        {!result && !isAnalyzing && !isTranscribing && (
          <View style={styles.voiceInteractionArea}>
            {/* Mode Toggle */}
            <View style={styles.modeToggle}>
              <TouchableOpacity
                style={[styles.modeBtn, inputMode === 'text' && styles.modeBtnActive]}
                onPress={() => setInputMode('text')}
              >
                <MaterialIcons name="edit" size={18} color={inputMode === 'text' ? Colors.warmWhite : Colors.textMuted} />
                <Text style={[styles.modeBtnText, inputMode === 'text' && styles.modeBtnTextActive]}>Type</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modeBtn, inputMode === 'voice' && styles.modeBtnActive]}
                onPress={() => setInputMode('voice')}
              >
                <MaterialIcons name="mic" size={18} color={inputMode === 'voice' ? Colors.warmWhite : Colors.textMuted} />
                <Text style={[styles.modeBtnText, inputMode === 'voice' && styles.modeBtnTextActive]}>Voice</Text>
              </TouchableOpacity>
            </View>

            {inputMode === 'voice' ? (
              <>
                {/* Mic Box */}
                <View style={styles.micGlowContainer}>
                  <TouchableOpacity
                    style={[styles.micButton, isRecording && { backgroundColor: '#FEE2E2' }]}
                    onPress={isRecording ? stopAndAnalyze : startRecording}
                    activeOpacity={0.85}
                    disabled={isTranscribing}
                  >
                    <MaterialIcons
                      name={isRecording ? 'stop' : 'mic'}
                      size={36}
                      color={isRecording ? '#EF4444' : Colors.sageGreen}
                    />
                  </TouchableOpacity>
                </View>

                {/* Recording duration */}
                {isRecording && (
                  <Text style={styles.durationText}>
                    {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')}
                  </Text>
                )}

                {/* Waveform */}
                <View style={styles.waveform}>
                  {waveAmplitudes.slice(0, 15).map((_amp, i) => (
                    <Animated.View
                      key={i}
                      style={[
                        styles.waveBar,
                        {
                          height: waveAnims[i],
                          backgroundColor: isRecording ? '#C4B5FD' : '#E5E7EB',
                          opacity: isRecording ? 1 : 0.5,
                        },
                      ]}
                    />
                  ))}
                </View>
                <Text style={styles.recordingActiveText}>
                  {isRecording ? '• RECORDING' : isTranscribing ? '• TRANSCRIBING...' : 'TAP TO RECORD'}
                </Text>

                {/* Transcript / Status Card */}
                <View style={styles.transcriptCard}>
                  <View style={styles.transcriptHeader}>
                    <View style={styles.liveIndicator}>
                      <View style={[styles.liveIndicatorDot, { backgroundColor: isRecording ? '#EF4444' : isTranscribing ? '#F59E0B' : Colors.warmGray400 }]} />
                      <Text style={[styles.liveIndicatorText, { color: isRecording ? '#EF4444' : isTranscribing ? '#F59E0B' : Colors.textMuted }]}>
                        {isRecording ? 'RECORDING AUDIO' : isTranscribing ? 'WHISPER TRANSCRIBING' : 'READY'}
                      </Text>
                    </View>
                    <View style={styles.privacyProtectedBadge}>
                      <Text style={styles.privacyProtectedText}>Encrypted</Text>
                    </View>
                  </View>
                  <Text style={styles.transcriptContent}>
                    {isTranscribing ? 'Sending audio to Whisper for transcription...'
                      : transcript || (isRecording ? 'Speak naturally about how you feel...' : 'Tap the microphone to start recording')}
                  </Text>
                </View>

                {/* Action Buttons */}
                {isRecording && (
                  <TouchableOpacity style={styles.endButton} onPress={stopAndAnalyze}>
                    <MaterialIcons name="check-circle-outline" size={20} color={Colors.warmWhite} />
                    <Text style={styles.endButtonText}>Stop & Analyze</Text>
                  </TouchableOpacity>
                )}

                {!isRecording && !isTranscribing && (
                  <View style={styles.voiceHintContainer}>
                    <MaterialIcons name="info-outline" size={14} color={Colors.textMuted} />
                    <Text style={styles.voiceHintText}>
                      Requires Whisper API key configured in Settings. Audio is recorded, sent to Whisper for transcription, then analyzed by the LLM.
                    </Text>
                  </View>
                )}
              </>
            ) : (
              <>
                {/* Text Input Mode */}
                <View style={styles.textInputIcon}>
                  <MaterialIcons name="chat-bubble-outline" size={36} color={Colors.violet} />
                </View>
                <Text style={styles.textInputPrompt}>How are you feeling today?</Text>
                <Text style={styles.textInputSubtext}>Share your thoughts — Seren will listen and respond with care.</Text>

                <View style={styles.transcriptCard}>
                  <View style={styles.transcriptHeader}>
                    <View style={styles.liveIndicator}>
                      <View style={[styles.liveIndicatorDot, { backgroundColor: textInput.trim() ? '#A78BFA' : Colors.warmGray400 }]} />
                      <Text style={[styles.liveIndicatorText, { color: textInput.trim() ? '#A78BFA' : Colors.textMuted }]}>YOUR THOUGHTS</Text>
                    </View>
                    <View style={styles.privacyProtectedBadge}>
                      <Text style={styles.privacyProtectedText}>Privacy Protected</Text>
                    </View>
                  </View>
                  <TextInput
                    style={styles.textInputField}
                    placeholder="I've been feeling..."
                    placeholderTextColor={Colors.textMuted}
                    multiline
                    value={textInput}
                    onChangeText={setTextInput}
                    textAlignVertical="top"
                  />
                </View>

                <TouchableOpacity
                  style={[styles.endButton, !textInput.trim() && { opacity: 0.5 }]}
                  onPress={() => { submitText(textInput); setTextInput(''); }}
                  disabled={!textInput.trim()}
                >
                  <MaterialIcons name="check-circle-outline" size={20} color={Colors.warmWhite} />
                  <Text style={styles.endButtonText}>Analyze</Text>
                </TouchableOpacity>
              </>
            )}

            {/* Footer */}
            <View style={styles.bottomFooter}>
              <View style={styles.e2eBadge}>
                <View style={[styles.insightDot, { backgroundColor: Colors.sageGreen }]} />
                <Text style={styles.e2eText}>END-TO-END ON-DEVICE NEURAL PROCESSING</Text>
              </View>
            </View>
          </View>
        )}

        {/* Transcribing / Analyzing */}
        {(isTranscribing || isAnalyzing) && (
          <View style={styles.analyzingContainer}>
            <Animated.View style={styles.analyzingIcon}>
              <MaterialIcons
                name={isTranscribing ? 'hearing' : 'psychology'}
                size={44}
                color={Colors.violet}
              />
            </Animated.View>
            <Text style={styles.analyzingText}>
              {isTranscribing ? 'Transcribing your voice...' : 'Analyzing your check-in...'}
            </Text>
            <Text style={styles.analyzingSubtext}>
              {isTranscribing
                ? 'Whisper AI is converting your speech to text'
                : 'Cross-referencing your words with biometric data'}
            </Text>
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
              {/* AI Empathetic Response */}
              {result?.empathyResponse ? (
                <View style={styles.empathySection}>
                  <View style={styles.empathyHeader}>
                    <MaterialIcons name="psychology" size={16} color={Colors.violet} />
                    <Text style={styles.empathyLabel}>Seren</Text>
                  </View>
                  <Text style={styles.empathyText}>{result.empathyResponse}</Text>
                </View>
              ) : null}

              {/* Follow-up question */}
              {result?.followUp ? (
                <View style={styles.followUpSection}>
                  <Text style={styles.followUpText}>{result.followUp}</Text>
                </View>
              ) : null}

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
    shadowColor: Colors.textPrimary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 15,
    elevation: 4,
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
    shadowColor: Colors.textPrimary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.04,
    shadowRadius: 12,
    elevation: 3,
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
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
    elevation: 6,
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
  empathySection: {
    backgroundColor: Colors.violet + '08',
    borderRadius: Radius.lg,
    padding: Spacing.md,
    marginBottom: Spacing.md,
    borderLeftWidth: 3,
    borderLeftColor: Colors.violet,
  },
  empathyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  empathyLabel: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    color: Colors.violet,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  empathyText: {
    fontSize: FontSize.sm,
    color: Colors.textPrimary,
    lineHeight: 22,
  },
  followUpSection: {
    backgroundColor: Colors.sageGreen + '10',
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginBottom: Spacing.md,
  },
  followUpText: {
    fontSize: FontSize.sm,
    color: Colors.sageGreenDark,
    lineHeight: 20,
    fontStyle: 'italic',
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
  modeToggle: {
    flexDirection: 'row',
    backgroundColor: Colors.warmWhite,
    borderRadius: Radius.full,
    padding: 4,
    gap: 4,
    borderWidth: 1,
    borderColor: '#F3F4F6',
    marginBottom: Spacing.md,
  },
  modeBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: Radius.full,
  },
  modeBtnActive: {
    backgroundColor: '#8B5CF6',
  },
  modeBtnText: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textMuted,
  },
  modeBtnTextActive: {
    color: Colors.warmWhite,
  },
  textInputIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#F5F3FF',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.sm,
  },
  textInputPrompt: {
    fontSize: FontSize.lg,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
    textAlign: 'center',
  },
  textInputSubtext: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
    textAlign: 'center',
    marginBottom: Spacing.sm,
  },
  textInputField: {
    fontSize: FontSize.md,
    color: Colors.textPrimary,
    lineHeight: 24,
    minHeight: 120,
    fontWeight: '500',
  },
  durationText: {
    fontSize: FontSize.xl,
    fontWeight: FontWeight.bold,
    color: '#EF4444',
    letterSpacing: 1,
  },
  voiceHintContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: Colors.softBlue + '10',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: Radius.lg,
    width: '100%',
    marginTop: Spacing.md,
  },
  voiceHintText: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    flex: 1,
    lineHeight: 18,
  },
});
