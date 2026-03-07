import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Image,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius, Shadow } from '@/constants/theme';
import { useWellness } from '@/hooks/useWellness';
import { GlassCard } from '@/components/ui/GlassCard';
import { useAlert } from '@/template';
import { useRouter } from 'expo-router';
import { scheduleMockHRVDropNotification } from '@/utils/notifications';

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { watchConnected, lastSyncTime, deleteAllData, exportData, requestHealthConnectPermissions, healthConnectAvailable, llmConfigured, llmProvider, whisperConfigured } = useWellness();
  const { showAlert } = useAlert();

  const [onDevice, setOnDevice] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [hrvAlerts, setHrvAlerts] = useState(true);
  const [sleepTracking, setSleepTracking] = useState(true);
  const [voicePrivacy, setVoicePrivacy] = useState(true);
  const [analytics, setAnalytics] = useState(false);

  const handlePurgeData = () => {
    showAlert(
      'Purge All Data',
      'This will permanently delete all your health data, check-ins, and insights. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Purge Everything',
          style: 'destructive',
          onPress: () => {
            deleteAllData();
            showAlert('Data Purged', 'All local data has been securely deleted.');
          },
        },
      ]
    );
  };

  const handleDisconnectWatch = () => {
    showAlert(
      'Disconnect Watch',
      'Disconnect Apple Watch Series 9 from Seren?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Disconnect', style: 'destructive', onPress: () => { } },
      ]
    );
  };

  const SettingRow = ({
    icon,
    iconColor,
    label,
    sublabel,
    value,
    onValueChange,
    showArrow,
    onPress,
    destructive,
  }: {
    icon: string;
    iconColor?: string;
    label: string;
    sublabel?: string;
    value?: boolean;
    onValueChange?: (v: boolean) => void;
    showArrow?: boolean;
    onPress?: () => void;
    destructive?: boolean;
  }) => (
    <TouchableOpacity
      style={styles.settingRow}
      onPress={onPress}
      disabled={!onPress && !onValueChange}
      activeOpacity={onPress ? 0.7 : 1}
    >
      <View style={[styles.settingIcon, { backgroundColor: (iconColor || Colors.violet) + '18' }]}>
        <MaterialIcons name={icon as any} size={18} color={iconColor || Colors.violet} />
      </View>
      <View style={styles.settingText}>
        <Text style={[styles.settingLabel, destructive && { color: Colors.error }]}>{label}</Text>
        {sublabel ? <Text style={styles.settingSubLabel}>{sublabel}</Text> : null}
      </View>
      {onValueChange !== undefined && value !== undefined ? (
        <Switch
          value={value}
          onValueChange={onValueChange}
          trackColor={{ false: Colors.warmGray300, true: Colors.sageGreenLight }}
          thumbColor={value ? Colors.sageGreen : Colors.warmWhite}
        />
      ) : null}
      {showArrow ? <MaterialIcons name="chevron-right" size={18} color={Colors.textMuted} /> : null}
    </TouchableOpacity>
  );

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
            <TouchableOpacity onPress={() => router.push('/(tabs)')} style={{ marginRight: 4 }}>
              <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
            </TouchableOpacity>
            <View style={styles.appIconContainer}>
              <Image source={require('@/assets/images/logo.png')} style={styles.appIcon} />
            </View>
            <Text style={styles.title}>Settings</Text>
          </View>
          <View style={styles.localBadge}>
            <MaterialIcons name="verified-user" size={14} color={Colors.sageGreenDark} />
            <Text style={styles.localText}>LOCAL</Text>
          </View>
        </View>

        {/* Connected Device */}
        <Text style={styles.sectionLabel}>Connected Device</Text>
        <GlassCard variant="default" style={styles.deviceCard}>
          <View style={styles.deviceHeader}>
            <View style={styles.deviceInfoContainer}>
              <View style={[styles.deviceIconBg, { backgroundColor: '#F0FDF4', borderColor: '#DCFCE7' }]}>
                <MaterialIcons name="watch" size={24} color={Colors.sageGreen} />
              </View>
              <View>
                <Text style={styles.deviceName}>Apple Watch Series 9</Text>
                <View style={styles.deviceStatusRow}>
                  <View style={styles.statusDot} />
                  <Text style={styles.deviceStatusText}>Connected</Text>
                </View>
              </View>
            </View>
            <View style={styles.batteryContainer}>
              <MaterialIcons name="battery-charging-full" size={20} color={Colors.sageGreen} />
              <Text style={styles.batteryText}>82%</Text>
            </View>
          </View>
          <View style={styles.rowDivider} />
          <View style={styles.syncRow}>
            <Text style={styles.syncText}>Last synced: Just now</Text>
            <TouchableOpacity style={styles.syncBtn}>
              <MaterialIcons name="sync" size={16} color={Colors.violet} />
              <Text style={styles.syncBtnText}>Sync Now</Text>
            </TouchableOpacity>
          </View>
        </GlassCard>

        <Text style={styles.sectionLabel}>Features</Text>
        <GlassCard variant="default" style={styles.settingsGroup}>
          <SettingRow
            icon="favorite"
            iconColor={Colors.sageGreen}
            label="Background HR Monitoring"
            sublabel={watchConnected ? 'Connected — HRV continuous' : 'Not connected'}
            value={onDevice}
            onValueChange={setOnDevice}
          />
          <View style={styles.rowDivider} />
          <SettingRow
            icon="bedtime"
            iconColor={Colors.softBlue}
            label="Sleep Analysis"
            sublabel="Auto-detect sleep stages"
            value={sleepTracking}
            onValueChange={setSleepTracking}
          />
          {healthConnectAvailable && !watchConnected ? (
            <>
              <View style={styles.rowDivider} />
              <SettingRow
                icon="sync"
                iconColor={Colors.violet}
                label="Connect Health Connect"
                sublabel="Grant sensor permissions"
                showArrow
                onPress={() => requestHealthConnectPermissions()}
              />
            </>
          ) : null}
        </GlassCard>

        {/* Data Governance */}
        <Text style={styles.sectionLabel}>Data Governance</Text>
        <GlassCard variant="default" style={styles.settingsGroup}>
          <SettingRow
            icon="security"
            iconColor={Colors.sageGreen}
            label="On-Device Processing"
            sublabel="Prevent cloud sync"
            value={voicePrivacy}
            onValueChange={setVoicePrivacy}
          />
          <View style={styles.rowDivider} />
          <SettingRow
            icon="analytics"
            iconColor={Colors.violet}
            label="Anonymous Analytics"
            sublabel="Help improve algorithms"
            value={analytics}
            onValueChange={setAnalytics}
          />
        </GlassCard>

        {/* AI Services */}
        <Text style={styles.sectionLabel}>AI Services</Text>
        <GlassCard variant="default" style={styles.settingsGroup}>
          {/* Status Row — LLM */}
          <View style={styles.settingRow}>
            <View style={[styles.settingIcon, { backgroundColor: Colors.violet + '18' }]}>
              <MaterialIcons name="psychology" size={18} color={Colors.violet} />
            </View>
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>Emotional Analysis (LLM)</Text>
              <Text style={styles.settingSubLabel}>
                {llmConfigured ? `Connected (${llmProvider})` : 'Not configured'}
              </Text>
            </View>
            <View style={[styles.llmStatusDot, { backgroundColor: llmConfigured ? Colors.sageGreen : Colors.warmGray300 }]} />
          </View>
          <View style={styles.rowDivider} />

          {/* Status Row — Whisper */}
          <View style={styles.settingRow}>
            <View style={[styles.settingIcon, { backgroundColor: Colors.softBlue + '18' }]}>
              <MaterialIcons name="hearing" size={18} color={Colors.softBlue} />
            </View>
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>Voice Transcription (Whisper)</Text>
              <Text style={styles.settingSubLabel}>
                {whisperConfigured ? 'Connected' : 'Not configured'}
              </Text>
            </View>
            <View style={[styles.llmStatusDot, { backgroundColor: whisperConfigured ? Colors.sageGreen : Colors.warmGray300 }]} />
          </View>
          <View style={styles.rowDivider} />

          {/* Config hint */}
          <View style={styles.apiKeySection}>
            <Text style={styles.apiKeyHint}>
              {llmConfigured
                ? `Provider: ${llmProvider} | Voice: ${whisperConfigured ? 'Whisper STT' : 'Text only'}\nAnalysis includes biometric cross-referencing with stress, sleep, and HRV data.`
                : 'AI services are configured in services/ai/aiConfig.ts.\nAdd your API key there to enable real voice transcription and deep emotional analysis.'}
            </Text>
          </View>
        </GlassCard>

        {/* Clinical Tools */}
        <Text style={styles.sectionLabel}>Clinical Tools</Text>
        <GlassCard variant="default" style={styles.settingsGroup}>
          <SettingRow
            icon="file-download"
            iconColor={Colors.violet}
            label="Export Health Data"
            sublabel="JSON format"
            showArrow
            onPress={async () => {
              showAlert('Export Data', 'Preparing your data export...');
              const path = await exportData();
              if (!path) showAlert('Export Failed', 'Could not export data. Make sure data is available.');
            }}
          />
          <View style={styles.rowDivider} />
          <SettingRow
            icon="delete-forever"
            iconColor={Colors.error}
            label="Delete All Data"
            sublabel="Permanent action"
            showArrow
            destructive
            onPress={handlePurgeData}
          />
        </GlassCard>

        {/* Developer / Debug */}
        <Text style={styles.sectionLabel}>Developer Tools</Text>
        <GlassCard variant="default" style={styles.settingsGroup}>
          <SettingRow
            icon="notifications-active"
            iconColor={Colors.violet}
            label="Simulate HRV Drop Alert"
            sublabel="Triggers in 5 seconds"
            showArrow
            onPress={async () => {
              const success = await scheduleMockHRVDropNotification();
              if (success) {
                showAlert('Notification Scheduled', 'Background the app now to see the push notification arrive in 5 seconds.');
              } else {
                showAlert('Permission Denied', 'Please enable notifications in system settings.');
              }
            }}
          />
        </GlassCard>
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
    paddingHorizontal: Spacing.lg,
    gap: Spacing.lg,
  },
  title: {
    fontSize: FontSize.lg,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
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
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    padding: Spacing.lg,
  },
  statusIcon: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: Colors.sageGreenMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusText: {
    flex: 1,
    gap: 4,
  },
  statusTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusTitle: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  activeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: Colors.sageGreenMuted,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: Radius.full,
  },
  activeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.sageGreen,
  },
  activeText: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    color: Colors.sageGreenDark,
  },
  statusSubtitle: {
    fontSize: FontSize.sm,
    color: Colors.sageGreenDark,
    lineHeight: 18,
  },
  sectionLabel: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    color: Colors.textMuted,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: -Spacing.sm,
  },
  settingsGroup: {
    overflow: 'hidden',
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    minHeight: 60,
  },
  settingIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingText: {
    flex: 1,
    gap: 2,
  },
  settingLabel: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  settingSubLabel: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    lineHeight: 16,
  },
  rowDivider: {
    height: 1,
    backgroundColor: Colors.warmGray100,
    marginLeft: 68,
  },
  watchSection: {
    padding: Spacing.md,
    gap: Spacing.md,
  },
  watchHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  watchName: {
    flex: 1,
    fontSize: FontSize.md,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  connectedDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  watchStats: {
    flexDirection: 'row',
    gap: Spacing.lg,
  },
  watchStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  watchStatText: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  disconnectBtn: {
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: Radius.md,
    borderWidth: 1.5,
    borderColor: Colors.error + '40',
    backgroundColor: Colors.errorMuted,
  },
  disconnectText: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.error,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: Spacing.md,
  },
  footerText: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  llmStatusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  apiKeySection: {
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  apiKeyHint: {
    fontSize: 11,
    color: Colors.textMuted,
    lineHeight: 16,
  },
  deviceCard: {
    padding: 0,
    marginBottom: Spacing.md,
    overflow: 'hidden',
  },
  deviceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: Spacing.md,
  },
  deviceInfoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  deviceIconBg: {
    width: 44,
    height: 44,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  deviceName: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    marginBottom: 2,
  },
  deviceStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.sageGreen,
  },
  deviceStatusText: {
    fontSize: FontSize.xs,
    color: Colors.sageGreenDark,
    fontWeight: '600',
  },
  batteryContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#F0FDF4',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.full,
  },
  batteryText: {
    fontSize: FontSize.xs,
    color: Colors.sageGreenDark,
    fontWeight: 'bold',
  },
  syncRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    backgroundColor: '#F9FAFB',
  },
  syncText: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
  },
  syncBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  syncBtnText: {
    fontSize: FontSize.xs,
    color: Colors.violet,
    fontWeight: 'bold',
  },
});
