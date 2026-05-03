import React, { useState, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Animated,
} from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { Image } from 'expo-image';

export default function SyncScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [scanning, setScanning] = useState(false);
    const [watchConnected, setWatchConnected] = useState(false);
    const scanAnim = useRef(new Animated.Value(0)).current;

    const handleScan = () => {
        setScanning(true);
        const anim = Animated.loop(
            Animated.sequence([
                Animated.timing(scanAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
                Animated.timing(scanAnim, { toValue: 0, duration: 1000, useNativeDriver: true }),
            ])
        );
        anim.start();

        // Simulate connection — actual Health Connect permissions
        // are granted later via Settings > Connect Health Connect
        setTimeout(() => {
            anim.stop();
            setScanning(false);
            setWatchConnected(true);
        }, 3000);
    };

    const handleNext = () => {
        router.push('/onboarding/privacy');
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
                            <Text style={styles.setupTitle}>Connect Device</Text>
                        </View>
                    </View>

                    <View style={styles.progressRow}>
                        <Text style={styles.stepText}>STEP 2 OF 3</Text>
                        <Text style={styles.stepSubText}>Connection</Text>
                    </View>
                    <View style={styles.progressBarBg}>
                        <View style={[styles.progressBarFill, { width: '66%' }]} />
                    </View>

                    <Text style={styles.heading}>Sync your smartwatch.</Text>
                    <Text style={styles.subtitle}>Seren uses your watch&apos;s sensors to proactively measure stress and sleep.</Text>
                </View>

                <View style={styles.syncContainer}>
                    <View style={styles.bluetoothCircleWrapper}>
                        <View style={styles.bluetoothCircle}>
                            <MaterialIcons name="bluetooth" size={32} color={Colors.violet} />
                        </View>
                    </View>
                    <Text style={styles.syncTitle}>Ready to Sync</Text>
                </View>

                {watchConnected ? (
                    <View style={styles.watchConnectedRow}>
                        <MaterialIcons name="check-circle" size={24} color={Colors.sageGreen} />
                        <View>
                            <Text style={styles.watchName}>Smartwatch Connected</Text>
                            <Text style={styles.watchSub}>Health Connect · Ready</Text>
                        </View>
                    </View>
                ) : (
                    <TouchableOpacity
                        style={[styles.scanButton, scanning && styles.scanButtonActive]}
                        onPress={handleScan}
                        disabled={scanning}
                    >
                        <MaterialIcons name="watch" size={20} color={Colors.warmWhite} style={{ marginRight: 8 }} />
                        <Text style={styles.scanButtonText}>
                            {scanning ? 'Connecting...' : 'Connect Smartwatch'}
                        </Text>
                    </TouchableOpacity>
                )}

                <TouchableOpacity onPress={handleNext} style={[styles.nextButton, watchConnected && styles.nextButtonActive]}>
                    <Text style={[styles.nextText, !watchConnected && { color: Colors.violet }]}>
                        {watchConnected ? 'Continue' : 'Skip for now'}
                    </Text>
                    <MaterialIcons name="arrow-forward" size={20} color={watchConnected ? Colors.warmWhite : Colors.violet} />
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
    syncContainer: {
        alignItems: 'center',
        marginBottom: Spacing.xxl,
        marginTop: Spacing.xl,
    },
    bluetoothCircleWrapper: {
        backgroundColor: Colors.warmWhite,
        width: 120,
        height: 120,
        borderRadius: 60,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.md,
    },
    bluetoothCircle: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: Colors.violetMuted,
        alignItems: 'center',
        justifyContent: 'center',
    },
    syncTitle: {
        fontSize: FontSize.lg,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    scanButton: {
        flexDirection: 'row',
        backgroundColor: Colors.sageGreen,
        paddingVertical: Spacing.lg,
        marginHorizontal: Spacing.xl,
        borderRadius: Radius.md,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.xl,
    },
    scanButtonActive: {
        backgroundColor: Colors.sageGreenDark,
    },
    scanButtonText: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
        color: Colors.warmWhite,
    },
    watchConnectedRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: Spacing.md,
        marginBottom: Spacing.xl,
        paddingHorizontal: Spacing.xl,
    },
    watchName: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.semibold,
        color: Colors.textPrimary,
    },
    watchSub: {
        fontSize: FontSize.sm,
        color: Colors.textMuted,
    },
    nextButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        marginHorizontal: Spacing.xl,
        paddingVertical: Spacing.md,
        borderRadius: Radius.full,
    },
    nextButtonActive: {
        backgroundColor: Colors.violet,
    },
    nextText: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
        color: Colors.warmWhite,
    },
});
