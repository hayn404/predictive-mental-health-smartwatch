import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from '@/components/ui/GlassCard';

export default function MeditationActivityScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();

    const TOTAL_SECONDS = 600; // 10 minutes
    const [isActive, setIsActive] = useState(false);
    const [secondsRemaining, setSecondsRemaining] = useState(TOTAL_SECONDS);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isActive && secondsRemaining > 0) {
            interval = setInterval(() => {
                setSecondsRemaining(prev => prev - 1);
            }, 1000);
        } else if (secondsRemaining === 0) {
            setIsActive(false);
        }
        return () => clearInterval(interval);
    }, [isActive, secondsRemaining]);

    const toggleTimer = () => {
        if (secondsRemaining === 0) {
            setSecondsRemaining(TOTAL_SECONDS);
        }
        setIsActive(!isActive);
    };

    const formatTime = (totalSeconds: number) => {
        const m = Math.floor(totalSeconds / 60);
        const s = totalSeconds % 60;
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    const currentSeconds = TOTAL_SECONDS - secondsRemaining;
    const progressPercent = (currentSeconds / TOTAL_SECONDS) * 100;

    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={[styles.header, { paddingTop: insets.top + Spacing.md }]}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Meditation</Text>
                <TouchableOpacity onPress={() => router.push('/(tabs)')} style={styles.closeBtn}>
                    <MaterialIcons name="close" size={24} color={Colors.textMuted} />
                </TouchableOpacity>
            </View>

            <View style={styles.content}>
                <GlassCard variant="default" style={styles.heroCard}>
                    <View style={[styles.iconBg, { backgroundColor: '#FFFBEB', borderColor: '#FEF3C7' }]}>
                        <MaterialIcons name="self-improvement" size={48} color={Colors.warning} />
                    </View>
                    <Text style={styles.title}>Mindful Presence</Text>
                    <Text style={styles.subtitle}>Let thoughts pass like clouds in the sky.</Text>
                </GlassCard>

                <View style={styles.playerContainer}>
                    <View style={styles.progressNav}>
                        <Text style={styles.timeText}>{formatTime(currentSeconds)}</Text>
                        <View style={styles.progressBarBg}>
                            <View style={[styles.progressBarFill, { width: `${progressPercent}%` }]} />
                        </View>
                        <Text style={styles.timeText}>- {formatTime(secondsRemaining)}</Text>
                    </View>
                    <TouchableOpacity style={styles.playBtn} onPress={toggleTimer}>
                        <MaterialIcons name={isActive ? "pause" : (secondsRemaining === 0 ? "refresh" : "play-arrow")} size={36} color={Colors.warmWhite} />
                    </TouchableOpacity>
                </View>

                <View style={styles.footer}>
                    <TouchableOpacity style={styles.doneBtn} onPress={() => router.back()}>
                        <Text style={styles.doneBtnText}>End Session</Text>
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
    title: {
        fontSize: FontSize.xxl,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
        marginBottom: Spacing.sm,
    },
    subtitle: {
        fontSize: FontSize.md,
        color: Colors.textSecondary,
        textAlign: 'center',
        lineHeight: 22,
    },
    playerContainer: {
        alignItems: 'center',
        marginTop: Spacing.xl,
    },
    progressNav: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: Spacing.md,
        marginBottom: Spacing.xl,
        width: '100%',
    },
    timeText: {
        fontSize: FontSize.xs,
        color: Colors.textMuted,
        fontVariant: ['tabular-nums'],
        width: 45,
        textAlign: 'center',
    },
    progressBarBg: {
        flex: 1,
        height: 6,
        backgroundColor: Colors.warmGray200,
        borderRadius: Radius.full,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        backgroundColor: Colors.warning,
        borderRadius: Radius.full,
    },
    playBtn: {
        width: 72,
        height: 72,
        borderRadius: 36,
        backgroundColor: Colors.warning,
        alignItems: 'center',
        justifyContent: 'center',
    },
    footer: {
        marginTop: 'auto',
        paddingTop: Spacing.xl,
    },
    doneBtn: {
        backgroundColor: '#F3F4F6',
        width: '100%',
        paddingVertical: Spacing.lg,
        borderRadius: Radius.full,
        alignItems: 'center',
    },
    doneBtnText: {
        color: Colors.textSecondary,
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
    },
});
