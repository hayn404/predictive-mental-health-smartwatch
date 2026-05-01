import React, { useState, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Dimensions,
    Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';

const { width } = Dimensions.get('window');

const slides = [
    {
        title: 'Your Mental\nWellness Companion',
        subtitle: 'Seren monitors stress, sleep, and anxiety in real-time — gently, privately, and proactively.',
        badge: 'Powered by AI',
        badgeIcon: 'psychology',
        heroIcon: 'psychology',
        colors: ['#FFFFFF', '#EAE5F5'] as const,
        accent: Colors.violet,
    },
    {
        title: 'Seamlessly\nConnected',
        subtitle: 'Sync with your smartwatch to capture heart rate, HRV, and sleep data for deep mental health insights.',
        badge: 'Smartwatch Sync',
        badgeIcon: 'watch',
        heroIcon: 'watch',
        colors: ['#FFFFFF', '#E3ECFA'] as const,
        accent: Colors.softBlue,
    },
    {
        title: 'Your Data\nStays Yours',
        subtitle: 'All processing happens on your device. We never sell, share, or upload your mental health data.',
        badge: '100% Private',
        badgeIcon: 'shield',
        heroIcon: 'verified-user',
        colors: ['#FFFFFF', '#E3F5EA'] as const,
        accent: Colors.sageGreen,
    },
];

export default function WelcomeScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [currentSlide, setCurrentSlide] = useState(0);
    const scrollRef = useRef<ScrollView>(null);
    const isLast = currentSlide === slides.length - 1;
    const active = slides[currentSlide];

    const goNext = () => {
        if (currentSlide < slides.length - 1) {
            const next = currentSlide + 1;
            setCurrentSlide(next);
            scrollRef.current?.scrollTo({ x: next * width, animated: true });
        } else {
            router.push('/onboarding/sync');
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={active.colors}
                style={StyleSheet.absoluteFill}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            />

            <View style={[styles.header, { paddingTop: insets.top + Spacing.xl }]}>
                <View style={styles.progressRow}>
                    <Text style={[styles.stepText, { color: active.accent }]}>STEP 1 OF 3</Text>
                    <Text style={styles.stepSubText}>Welcome</Text>
                </View>
                <View style={styles.progressBarBg}>
                    <View style={[styles.progressBarFill, { width: '33%', backgroundColor: active.accent }]} />
                </View>
            </View>

            <ScrollView
                ref={scrollRef}
                horizontal
                pagingEnabled
                showsHorizontalScrollIndicator={false}
                scrollEnabled={false}
                style={{ flex: 1 }}
            >
                {slides.map((slide, i) => (
                    <View key={i} style={[styles.slide, { width }]}>
                        <View style={styles.heroIconContainer}>
                            <View style={[styles.heroIconCircle, { backgroundColor: slide.accent + '22', borderColor: slide.accent + '55' }]}>
                                {i === 0 ? (
                                    <Image
                                        source={require('@/assets/images/seren-brain.png')}
                                        style={styles.heroBrainImage}
                                        resizeMode="contain"
                                    />
                                ) : (
                                    <MaterialIcons name={slide.heroIcon as any} size={72} color={slide.accent} />
                                )}
                            </View>
                        </View>
                        <View style={[styles.slideContent, { paddingBottom: insets.bottom + 180 }]}>
                            <View style={styles.badgeRow}>
                                <MaterialIcons name={slide.badgeIcon as any} size={14} color={slide.accent} />
                                <Text style={styles.badge}>{slide.badge}</Text>
                            </View>
                            <Text style={styles.slideTitle}>{slide.title}</Text>
                            <Text style={styles.slideSubtitle}>{slide.subtitle}</Text>
                        </View>
                    </View>
                ))}
            </ScrollView>

            {/* Bottom Controls */}
            <View style={[styles.bottomControls, { paddingBottom: insets.bottom + Spacing.lg }]}>
                <View style={styles.dots}>
                    {slides.map((_, i) => (
                        <View
                            key={i}
                            style={[
                                styles.dot,
                                i === currentSlide
                                    ? { ...styles.dotActive, backgroundColor: active.accent }
                                    : styles.dotInactive,
                            ]}
                        />
                    ))}
                </View>

                <TouchableOpacity
                    style={[styles.nextButton, { backgroundColor: active.accent }]}
                    onPress={goNext}
                >
                    <Text style={styles.nextText}>{isLast ? 'Continue Setup' : 'Continue'}</Text>
                    <MaterialIcons name="arrow-forward" size={18} color={Colors.warmWhite} />
                </TouchableOpacity>

                {currentSlide < slides.length - 1 ? (
                    <TouchableOpacity onPress={() => router.push('/onboarding/sync')} style={styles.skipBtn}>
                        <Text style={styles.skipText}>Skip introduction</Text>
                    </TouchableOpacity>
                ) : <View style={styles.skipBtnPlaceholder} />}
            </View>
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
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 10,
    },
    progressRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: Spacing.sm,
    },
    stepText: {
        fontSize: FontSize.xs,
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
        backgroundColor: 'rgba(0,0,0,0.08)',
        borderRadius: Radius.full,
        marginBottom: Spacing.xl,
    },
    progressBarFill: {
        height: 4,
        borderRadius: Radius.full,
    },
    slide: {
        flex: 1,
        position: 'relative',
    },
    heroIconContainer: {
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingTop: 140,
    },
    heroIconCircle: {
        width: 180,
        height: 180,
        borderRadius: 90,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
    },
    heroBrainImage: {
        width: 120,
        height: 120,
    },
    slideContent: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: Spacing.xl,
    },
    badgeRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(255,255,255,0.7)',
        alignSelf: 'flex-start',
        paddingHorizontal: Spacing.md,
        paddingVertical: 6,
        borderRadius: Radius.full,
        marginBottom: Spacing.md,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.08)',
    },
    badge: {
        fontSize: FontSize.xs,
        color: Colors.textPrimary,
        fontWeight: FontWeight.semibold,
    },
    slideTitle: {
        fontSize: FontSize.xxl,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
        lineHeight: 38,
        marginBottom: Spacing.md,
    },
    slideSubtitle: {
        fontSize: FontSize.md,
        color: Colors.textSecondary,
        lineHeight: 26,
        fontWeight: FontWeight.regular,
    },
    bottomControls: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        alignItems: 'center',
        gap: Spacing.md,
        paddingHorizontal: Spacing.xl,
        paddingTop: Spacing.lg,
    },
    dots: {
        flexDirection: 'row',
        gap: 8,
    },
    dot: {
        height: 8,
        borderRadius: 4,
    },
    dotActive: {
        width: 28,
    },
    dotInactive: {
        width: 8,
        backgroundColor: 'rgba(0,0,0,0.15)',
    },
    nextButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        paddingVertical: Spacing.md,
        paddingHorizontal: Spacing.xl,
        borderRadius: Radius.full,
        width: '100%',
    },
    nextText: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.semibold,
        color: Colors.warmWhite,
    },
    skipBtn: {
        paddingVertical: 8,
    },
    skipText: {
        fontSize: FontSize.sm,
        color: Colors.textMuted,
        fontWeight: FontWeight.medium,
    },
    skipBtnPlaceholder: {
        paddingVertical: 8,
        height: 36,
    },
});
