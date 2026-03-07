import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated, Easing } from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from '@/components/ui/GlassCard';

type Phase = 'Inhale' | 'Hold' | 'Exhale' | 'HoldOut';

export default function BreathingActivityScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [isActive, setIsActive] = useState(false);
    const [phase, setPhase] = useState<Phase>('Inhale');

    const scaleAnim = useRef(new Animated.Value(1)).current;
    const opacityAnim = useRef(new Animated.Value(0.2)).current;

    // Clean up any remaining animations on unmount
    useEffect(() => {
        return () => {
            scaleAnim.stopAnimation();
            opacityAnim.stopAnimation();
        };
    }, []);

    const startBreathingCycle = () => {
        // Phase 1: Inhale (4s)
        setPhase('Inhale');
        Animated.parallel([
            Animated.timing(scaleAnim, {
                toValue: 2,
                duration: 4000,
                useNativeDriver: true,
                easing: Easing.inOut(Easing.ease),
            }),
            Animated.timing(opacityAnim, {
                toValue: 0.8,
                duration: 4000,
                useNativeDriver: true,
            })
        ]).start(({ finished }) => {
            if (!finished || !isActive) return;

            // Phase 2: Hold (4s)
            setPhase('Hold');
            setTimeout(() => {
                if (!isActive) return;

                // Phase 3: Exhale (4s)
                setPhase('Exhale');
                Animated.parallel([
                    Animated.timing(scaleAnim, {
                        toValue: 1,
                        duration: 4000,
                        useNativeDriver: true,
                        easing: Easing.inOut(Easing.ease),
                    }),
                    Animated.timing(opacityAnim, {
                        toValue: 0.2,
                        duration: 4000,
                        useNativeDriver: true,
                    })
                ]).start(({ finished }) => {
                    if (!finished || !isActive) return;

                    // Phase 4: Hold Out (4s)
                    setPhase('HoldOut');
                    setTimeout(() => {
                        if (!isActive) return;
                        startBreathingCycle(); // Loop
                    }, 4000);
                });
            }, 4000);
        });
    };

    useEffect(() => {
        if (isActive) {
            startBreathingCycle();
        } else {
            scaleAnim.stopAnimation();
            opacityAnim.stopAnimation();
            Animated.parallel([
                Animated.timing(scaleAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
                Animated.timing(opacityAnim, { toValue: 0.2, duration: 500, useNativeDriver: true }),
            ]).start();
            setPhase('Inhale');
        }
    }, [isActive]);

    const toggleSession = () => {
        setIsActive(!isActive);
    };

    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={[styles.header, { paddingTop: insets.top + Spacing.md }]}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Breathing</Text>
                <TouchableOpacity onPress={() => router.push('/(tabs)')} style={styles.closeBtn}>
                    <MaterialIcons name="close" size={24} color={Colors.textMuted} />
                </TouchableOpacity>
            </View>

            <View style={styles.content}>

                <View style={styles.animationContainer}>
                    <Animated.View style={[styles.breathingCircleBg, { transform: [{ scale: scaleAnim }], opacity: opacityAnim }]} />
                    <View style={styles.breathingCircleInner}>
                        {isActive ? (
                            <Text style={styles.phaseText}>
                                {phase === 'Inhale' && 'Inhale...'}
                                {phase === 'Hold' && 'Hold...'}
                                {phase === 'Exhale' && 'Exhale...'}
                                {phase === 'HoldOut' && 'Hold...'}
                            </Text>
                        ) : (
                            <MaterialIcons name="air" size={48} color={Colors.softBlue} />
                        )}
                    </View>
                </View>

                {!isActive && (
                    <View style={styles.instructions}>
                        <Text style={styles.instructionTitle}>Box Breathing</Text>
                        <Text style={styles.instructionStep}>1. Inhale for 4s</Text>
                        <Text style={styles.instructionStep}>2. Hold for 4s</Text>
                        <Text style={styles.instructionStep}>3. Exhale for 4s</Text>
                        <Text style={styles.instructionStep}>4. Hold empty for 4s</Text>
                    </View>
                )}

                <View style={styles.footer}>
                    <TouchableOpacity style={[styles.startBtn, isActive && styles.stopBtn]} onPress={toggleSession}>
                        <Text style={[styles.startBtnText, isActive && styles.stopBtnText]}>
                            {isActive ? 'Stop Session' : 'Start Session'}
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
        backgroundColor: Colors.warmWhite, // User requested white background
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
        alignItems: 'center',
        justifyContent: 'center',
    },
    animationContainer: {
        width: 250,
        height: 250,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.xxl,
    },
    breathingCircleBg: {
        position: 'absolute',
        width: 120,
        height: 120,
        borderRadius: 60,
        backgroundColor: Colors.softBlue,
    },
    breathingCircleInner: {
        width: 120,
        height: 120,
        borderRadius: 60,
        backgroundColor: Colors.warmWhite,
        borderWidth: 2,
        borderColor: '#E0F2FE',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
    },
    phaseText: {
        fontSize: FontSize.lg,
        fontWeight: 'bold',
        color: Colors.softBlue,
    },
    instructions: {
        padding: Spacing.xl,
        gap: Spacing.sm,
        alignItems: 'center',
    },
    instructionTitle: {
        fontSize: FontSize.xl,
        fontWeight: 'bold',
        color: Colors.textPrimary,
        marginBottom: Spacing.sm,
    },
    instructionStep: {
        fontSize: FontSize.md,
        fontWeight: '500',
        color: Colors.textSecondary,
        textAlign: 'center',
    },
    footer: {
        marginTop: 'auto',
        width: '100%',
        paddingTop: Spacing.xl,
    },
    startBtn: {
        backgroundColor: Colors.softBlue,
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
        backgroundColor: '#FEE2E2',
    },
    stopBtnText: {
        color: '#EF4444',
    }
});
