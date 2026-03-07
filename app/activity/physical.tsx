import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from '@/components/ui/GlassCard';

export default function PhysicalActivityScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();

    const [isActive, setIsActive] = useState(false);
    const [secondsElapsed, setSecondsElapsed] = useState(0);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isActive) {
            interval = setInterval(() => {
                setSecondsElapsed(prev => prev + 1);
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isActive]);

    const toggleTimer = () => {
        if (isActive) {
            // Prompt to save
            Alert.alert(
                "End Activity?",
                `You've been active for ${formatTime(secondsElapsed)}. Save this session?`,
                [
                    { text: "Continue", style: "cancel" },
                    {
                        text: "Save & End", onPress: () => {
                            setIsActive(false);
                            router.back();
                        }
                    }
                ]
            );
        } else {
            setIsActive(true);
        }
    };

    const formatTime = (totalSeconds: number) => {
        const m = Math.floor(totalSeconds / 60);
        const s = totalSeconds % 60;
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    // Estimate calories burned (rough estimate: 4 calories per minute)
    const calories = Math.floor(secondsElapsed / 60 * 4);

    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={[styles.header, { paddingTop: insets.top + Spacing.md }]}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Physical</Text>
                <TouchableOpacity onPress={() => router.push('/(tabs)')} style={styles.closeBtn}>
                    <MaterialIcons name="close" size={24} color={Colors.textMuted} />
                </TouchableOpacity>
            </View>

            <View style={styles.content}>
                <GlassCard variant="default" style={styles.heroCard}>
                    <View style={[styles.iconBg, { backgroundColor: '#F0FDF4', borderColor: '#DCFCE7' }]}>
                        <MaterialIcons name="directions-run" size={48} color={Colors.sageGreen} />
                    </View>
                    <Text style={styles.timerLarge}>{formatTime(secondsElapsed)}</Text>
                    <Text style={styles.subtitle}>{isActive ? 'Keep moving! Tracking...' : 'Ready for a brisk walk?'}</Text>
                </GlassCard>

                <View style={styles.statsContainer}>
                    <View style={styles.statBox}>
                        <MaterialIcons name="schedule" size={24} color={Colors.textSecondary} />
                        <Text style={styles.statValue}>15:00</Text>
                        <Text style={styles.statLabel}>Target</Text>
                    </View>

                    <View style={styles.statBox}>
                        <MaterialIcons name="local-fire-department" size={24} color={Colors.textSecondary} />
                        <Text style={styles.statValue}>{calories}</Text>
                        <Text style={styles.statLabel}>Calories Burned</Text>
                    </View>
                </View>

                <View style={styles.footer}>
                    <TouchableOpacity
                        style={[styles.startBtn, isActive && styles.stopBtn]}
                        onPress={toggleTimer}
                    >
                        <Text style={[styles.startBtnText, isActive && styles.stopBtnText]}>
                            {isActive ? 'Pause & End' : 'Start Activity'}
                        </Text>
                    </TouchableOpacity>
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.warmWhite, // Changed to white
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: Spacing.md,
        paddingBottom: Spacing.md,
        backgroundColor: Colors.warmWhite,
        borderBottomWidth: 1,
        borderBottomColor: Colors.warmGray200,
    },
    backBtn: { padding: 4 },
    closeBtn: { padding: 4 },
    headerTitle: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    content: {
        flex: 1,
        padding: Spacing.lg,
        justifyContent: 'center',
    },
    heroCard: {
        alignItems: 'center',
        padding: Spacing.xxl,
        marginBottom: Spacing.xl,
        backgroundColor: '#FAFAFA',
        borderWidth: 1,
        borderColor: '#E5E7EB',
    },
    iconBg: {
        width: 96,
        height: 96,
        borderRadius: 48,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.xl,
        borderWidth: 1,
    },
    timerLarge: {
        fontSize: 48,
        fontWeight: 'bold',
        color: Colors.textPrimary,
        fontVariant: ['tabular-nums'],
        marginBottom: Spacing.xs,
    },
    subtitle: {
        fontSize: FontSize.md,
        color: Colors.textSecondary,
        textAlign: 'center',
    },
    statsContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        gap: Spacing.md,
    },
    statBox: {
        flex: 1,
        backgroundColor: '#FAFAFA',
        padding: Spacing.xl,
        borderRadius: Radius.lg,
        alignItems: 'center',
        gap: 8,
        borderWidth: 1,
        borderColor: '#E5E7EB',
    },
    statValue: {
        fontSize: FontSize.xl,
        fontWeight: 'bold',
        color: Colors.textPrimary,
    },
    statLabel: {
        fontSize: FontSize.xs,
        color: Colors.textMuted,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    footer: {
        marginTop: 'auto',
        paddingTop: Spacing.xl,
    },
    startBtn: {
        backgroundColor: Colors.sageGreen,
        width: '100%',
        paddingVertical: Spacing.lg,
        borderRadius: Radius.full,
        alignItems: 'center',
    },
    startBtnText: {
        color: Colors.warmWhite,
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
    },
    stopBtn: {
        backgroundColor: '#FEF2F2',
        borderWidth: 1,
        borderColor: '#FCA5A5',
    },
    stopBtnText: {
        color: '#EF4444',
    }
});
