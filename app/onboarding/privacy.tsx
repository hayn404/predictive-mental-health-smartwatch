import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Switch,
} from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { Image } from 'expo-image';
import { requestNotificationPermissions } from '@/utils/notifications';

export default function PrivacyScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [onDevice, setOnDevice] = useState(true);

    const handleFinish = async () => {
        // Request notification permissions explicitly before continuing
        await requestNotificationPermissions();
        router.replace('/(tabs)');
    };

    return (
        <View style={styles.container}>
            <ScrollView
                showsVerticalScrollIndicator={false}
                contentContainerStyle={{ paddingTop: insets.top + Spacing.xl, paddingBottom: insets.bottom + Spacing.xl }}
            >
                <View style={styles.header}>
                    <View style={styles.serenBadgeRow}>
                        <View style={styles.titleRow}>
                            <View style={styles.appIconContainer}>
                                <Image source={require('@/assets/images/seren-brain.png')} style={styles.appIcon} />
                            </View>
                            <Text style={styles.setupTitle}>Privacy & Permissions</Text>
                        </View>
                    </View>

                    <View style={styles.progressRow}>
                        <Text style={styles.stepText}>STEP 3 OF 3</Text>
                        <Text style={styles.stepSubText}>Setup Complete</Text>
                    </View>
                    <View style={styles.progressBarBg}>
                        <View style={[styles.progressBarFill, { width: '100%' }]} />
                    </View>

                    <Text style={styles.heading}>Your data stays yours.</Text>
                    <Text style={styles.subtitle}>We take a radically different approach to health data. Everything happens right here on your phone.</Text>
                </View>

                <View style={styles.encryptionBox}>
                    <View style={styles.encryptionIconWrapper}>
                        <MaterialIcons name="shield" size={24} color={Colors.sageGreen} />
                    </View>
                    <View style={styles.encryptionTextBox}>
                        <Text style={styles.encryptionTitle}>Keep all data on-device</Text>
                        <Text style={styles.encryptionDesc}>When active, none of your health metrics or voice check-ins are sent to our servers.</Text>
                    </View>
                    <Switch
                        value={onDevice}
                        onValueChange={setOnDevice}
                        trackColor={{ false: Colors.warmGray300, true: Colors.sageGreenLight }}
                        thumbColor={onDevice ? Colors.sageGreen : Colors.warmWhite}
                    />
                </View>

                <View style={styles.permissionsContainer}>
                    <Text style={styles.permissionsHeader}>REQUIRED PERMISSIONS</Text>
                    <View style={styles.permissionsList}>
                        <View style={styles.permissionCard}>
                            <View style={styles.permIconWrapperRed}>
                                <MaterialIcons name="show-chart" size={20} color={Colors.error} />
                            </View>
                            <View style={styles.permTextContainer}>
                                <Text style={styles.permLabel}>Heart Rate & HRV</Text>
                                <Text style={styles.permDesc}>Allows us to measure physiological stress in real-time.</Text>
                            </View>
                            <MaterialIcons name="check-circle" size={20} color={Colors.sageGreen} />
                        </View>

                        <View style={styles.permissionCard}>
                            <View style={styles.permIconWrapperBlue}>
                                <MaterialIcons name="bedtime" size={20} color={Colors.softBlue} />
                            </View>
                            <View style={styles.permTextContainer}>
                                <Text style={styles.permLabel}>Sleep Analysis</Text>
                                <Text style={styles.permDesc}>Monitors recovery and neural resting periods overnight.</Text>
                            </View>
                            <MaterialIcons name="check-circle" size={20} color={Colors.sageGreen} />
                        </View>

                        <View style={styles.permissionCard}>
                            <View style={styles.permIconWrapperViolet}>
                                <MaterialIcons name="notifications-active" size={20} color={Colors.violet} />
                            </View>
                            <View style={styles.permTextContainer}>
                                <Text style={styles.permLabel}>Proactive Alerts</Text>
                                <Text style={styles.permDesc}>Notifies you directly if your stress levels or HRV drops.</Text>
                            </View>
                            <MaterialIcons name="check-circle" size={20} color={Colors.sageGreen} />
                        </View>
                    </View>
                </View>

                <TouchableOpacity
                    style={styles.startButton}
                    onPress={handleFinish}
                >
                    <Text style={styles.startButtonText}>
                        Start Your Journey
                    </Text>
                    <MaterialIcons name="arrow-forward" size={20} color={Colors.warmWhite} />
                </TouchableOpacity>

                <Text style={styles.footerText}>PRIVACY-FIRST MENTAL WELLNESS • V1.0.4</Text>
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.cream,
    },
    header: {
        marginBottom: Spacing.xl,
        paddingHorizontal: Spacing.xl,
    },
    serenBadgeRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: Spacing.lg,
    },
    titleRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    appIconContainer: {
        width: 32,
        height: 32,
        borderRadius: Radius.md,
        overflow: 'hidden',
        backgroundColor: Colors.warmWhite,
    },
    appIcon: {
        width: '100%',
        height: '100%',
        resizeMode: 'contain',
    },
    setupTitle: {
        fontSize: FontSize.lg,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    progressRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: Spacing.sm,
    },
    stepText: {
        fontSize: FontSize.xs,
        color: Colors.violet,
        fontWeight: '700',
        letterSpacing: 1.2,
    },
    stepSubText: {
        fontSize: FontSize.xs,
        color: Colors.textSecondary,
        fontWeight: '500',
    },
    progressBarBg: {
        height: 4,
        backgroundColor: Colors.violetMuted,
        borderRadius: Radius.full,
        marginBottom: Spacing.xl,
    },
    progressBarFill: {
        height: 4,
        backgroundColor: Colors.violet,
        borderRadius: Radius.full,
    },
    heading: {
        fontSize: FontSize.xxl,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
        marginBottom: Spacing.xs,
    },
    subtitle: {
        fontSize: FontSize.md,
        color: Colors.textSecondary,
        lineHeight: 24,
    },
    encryptionBox: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: Spacing.md,
        backgroundColor: '#F0FDF4',
        marginHorizontal: Spacing.xl,
        padding: Spacing.lg,
        borderRadius: 20,
        marginBottom: Spacing.xxl,
        borderWidth: 1,
        borderColor: '#DCFCE7',
    },
    encryptionIconWrapper: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: Colors.warmWhite,
        alignItems: 'center',
        justifyContent: 'center',
    },
    encryptionTextBox: {
        flex: 1,
    },
    encryptionTitle: {
        fontSize: FontSize.md,
        fontWeight: 'bold',
        color: Colors.textPrimary,
        marginBottom: 4,
    },
    encryptionDesc: {
        fontSize: FontSize.xs,
        color: Colors.sageGreenDark,
        lineHeight: 18,
    },
    permissionsContainer: {
        marginHorizontal: Spacing.xl,
        marginBottom: Spacing.xxl,
    },
    permissionsHeader: {
        fontSize: FontSize.xs,
        fontWeight: 'bold',
        color: Colors.textMuted,
        letterSpacing: 1,
        marginBottom: Spacing.md,
    },
    permissionsList: {
        gap: Spacing.sm,
    },
    permissionCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: Colors.warmWhite,
        padding: Spacing.md,
        borderRadius: Radius.lg,
        gap: Spacing.md,
        borderWidth: 1,
        borderColor: Colors.warmGray100,
    },
    permIconWrapperRed: {
        width: 40,
        height: 40,
        borderRadius: Radius.full,
        backgroundColor: '#FEF2F2',
        alignItems: 'center',
        justifyContent: 'center',
    },
    permIconWrapperBlue: {
        width: 40,
        height: 40,
        borderRadius: Radius.full,
        backgroundColor: '#F5F3FF',
        alignItems: 'center',
        justifyContent: 'center',
    },
    permIconWrapperViolet: {
        width: 40,
        height: 40,
        borderRadius: Radius.full,
        backgroundColor: Colors.violetMuted,
        alignItems: 'center',
        justifyContent: 'center',
    },
    permTextContainer: {
        flex: 1,
        gap: 2,
    },
    permLabel: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.semibold,
        color: Colors.textPrimary,
    },
    permDesc: {
        fontSize: FontSize.xs,
        color: Colors.textSecondary,
        lineHeight: 16,
    },
    startButton: {
        flexDirection: 'row',
        backgroundColor: Colors.violet,
        marginHorizontal: Spacing.xl,
        paddingVertical: Spacing.lg,
        borderRadius: Radius.full,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.xl,
        gap: 8,
    },
    startButtonText: {
        fontSize: FontSize.md,
        fontWeight: 'bold',
        color: Colors.warmWhite,
    },
    footerText: {
        textAlign: 'center',
        fontSize: 10,
        fontWeight: 'bold',
        color: Colors.textMuted,
        letterSpacing: 1,
        marginTop: Spacing.xs,
    },
});
