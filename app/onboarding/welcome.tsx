import React, { useState, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Dimensions,
} from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';

const { width } = Dimensions.get('window');

const slides = [
    {
        image: require('@/assets/images/onboarding1.png'),
        title: 'Your Mental\nWellness Companion',
        subtitle: 'Seren monitors stress, sleep, and anxiety in real-time — gently, privately, and proactively.',
        badge: 'Powered by AI',
        badgeIcon: 'psychology',
    },
    {
        image: require('@/assets/images/onboarding2.png'),
        title: 'Seamlessly\nConnected',
        subtitle: 'Sync with your smartwatch to capture heart rate, HRV, and sleep data for deep mental health insights.',
        badge: 'Smartwatch Sync',
        badgeIcon: 'watch',
    },
    {
        image: require('@/assets/images/onboarding3.png'),
        title: 'Your Data\nStays Yours',
        subtitle: 'All processing happens on your device. We never sell, share, or upload your mental health data.',
        badge: '100% Private',
        badgeIcon: 'shield',
    },
];

export default function WelcomeScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [currentSlide, setCurrentSlide] = useState(0);
    const scrollRef = useRef<ScrollView>(null);
    const isLast = currentSlide === slides.length - 1;

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
            <View style={[styles.header, { paddingTop: insets.top + Spacing.xl }]}>
                <View style={styles.progressRow}>
                    <Text style={styles.stepText}>STEP 1 OF 3</Text>
                    <Text style={styles.stepSubText}>Welcome</Text>
                </View>
                <View style={styles.progressBarBg}>
                    <View style={[styles.progressBarFill, { width: '33%' }]} />
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
                        <Image
                            source={slide.image}
                            style={styles.slideImage}
                            contentFit="cover"
                            transition={400}
                        />
                        <View style={styles.slideOverlay} />
                        <View style={[styles.slideContent, { paddingBottom: insets.bottom + 180 }]}>
                            <View style={styles.badgeRow}>
                                <MaterialIcons name={slide.badgeIcon as any} size={14} color={Colors.sageGreen} />
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
                                i === currentSlide ? styles.dotActive : styles.dotInactive,
                            ]}
                        />
                    ))}
                </View>

                <TouchableOpacity style={styles.nextButton} onPress={goNext}>
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
        backgroundColor: '#252240',
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
        color: Colors.violet,
        fontWeight: '700',
        letterSpacing: 1.2,
    },
    stepSubText: {
        fontSize: FontSize.xs,
        color: Colors.warmWhite,
        fontWeight: '500',
    },
    progressBarBg: {
        height: 4,
        backgroundColor: 'rgba(255,255,255,0.2)',
        borderRadius: Radius.full,
        marginBottom: Spacing.xl,
    },
    progressBarFill: {
        height: 4,
        backgroundColor: Colors.violet,
        borderRadius: Radius.full,
    },
    slide: {
        flex: 1,
        position: 'relative',
    },
    slideImage: {
        ...StyleSheet.absoluteFillObject,
    },
    slideOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(37, 34, 64, 0.30)',
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
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignSelf: 'flex-start',
        paddingHorizontal: Spacing.md,
        paddingVertical: 6,
        borderRadius: Radius.full,
        marginBottom: Spacing.md,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.3)',
    },
    badge: {
        fontSize: FontSize.xs,
        color: Colors.warmWhite,
        fontWeight: FontWeight.semibold,
    },
    slideTitle: {
        fontSize: FontSize.xxl,
        fontWeight: FontWeight.bold,
        color: Colors.warmWhite,
        lineHeight: 38,
        marginBottom: Spacing.md,
    },
    slideSubtitle: {
        fontSize: FontSize.md,
        color: 'rgba(255,255,255,0.85)',
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
        backgroundColor: Colors.sageGreen,
    },
    dotInactive: {
        width: 8,
        backgroundColor: 'rgba(255,255,255,0.4)',
    },
    nextButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: Colors.violet,
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
        color: 'rgba(255,255,255,0.7)',
        fontWeight: FontWeight.medium,
    },
    skipBtnPlaceholder: {
        paddingVertical: 8,
        height: 36, // To keep the layout stable when the button disappears
    },
});
