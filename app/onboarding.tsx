import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
  Dimensions,
  Switch,
  Platform,
} from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';

const { width, height } = Dimensions.get('window');

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

export default function Onboarding() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [currentSlide, setCurrentSlide] = useState(0);
  const [onDevice, setOnDevice] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [watchConnected, setWatchConnected] = useState(false);
  const scrollRef = useRef<ScrollView>(null);
  const scanAnim = useRef(new Animated.Value(0)).current;
  const isLast = currentSlide === slides.length - 1;
  const isSetup = currentSlide === slides.length; // setup page

  const goNext = () => {
    if (currentSlide < slides.length - 1) {
      const next = currentSlide + 1;
      setCurrentSlide(next);
      scrollRef.current?.scrollTo({ x: next * width, animated: true });
    } else {
      setCurrentSlide(slides.length); // go to setup
    }
  };

  const handleScan = () => {
    setScanning(true);
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
        Animated.timing(scanAnim, { toValue: 0, duration: 1000, useNativeDriver: true }),
      ])
    );
    anim.start();
    setTimeout(() => {
      anim.stop();
      setScanning(false);
      setWatchConnected(true);
    }, 3000);
  };

  const handleGetStarted = () => {
    router.replace('/(tabs)');
  };

  if (isSetup) {
    return (
      <ScrollView
        showsVerticalScrollIndicator={false}
        style={styles.setupContainer}
        contentContainerStyle={{ paddingTop: insets.top + Spacing.xl, paddingBottom: insets.bottom + Spacing.xl }}
      >
        <View style={styles.setupHeader}>
          <View style={styles.serenBadgeRow}>
            <View style={styles.titleRow}>
              <View style={styles.appIconContainer}>
                <Image source={require('@/assets/images/logo.png')} style={styles.appIcon} />
              </View>
              <Text style={styles.setupTitle}>Setup Seren</Text>
            </View>
            <View style={styles.localBadge}>
              <MaterialIcons name="verified-user" size={14} color={Colors.sageGreen} />
              <Text style={styles.localText}>LOCAL</Text>
            </View>
          </View>

          <View style={styles.progressRow}>
            <Text style={styles.stepText}>STEP 1 OF 3</Text>
            <Text style={styles.stepSubText}>Device Sync</Text>
          </View>
          <View style={styles.progressBarBg}>
            <View style={[styles.progressBarFill, { width: '33%' }]} />
          </View>

          <Text style={styles.setupHeading}>Personalized for you.</Text>
          <Text style={styles.setupSubtitle}>Understand your mind through your body's data.</Text>
        </View>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.featureCardsContainer}>
          <View style={[styles.featureCard, { backgroundColor: '#F0FDF4' }]}>
            <View style={[styles.featureIconBg, { borderColor: '#DCFCE7' }]}>
              <MaterialIcons name="verified-user" size={20} color={Colors.sageGreen} />
            </View>
            <Text style={styles.featureTitle}>Privacy First</Text>
            <Text style={styles.featureDesc}>All health data is processed 100% on your device. We never see it.</Text>
          </View>
          <View style={[styles.featureCard, { backgroundColor: '#F0F9FF' }]}>
            <View style={[styles.featureIconBg, { borderColor: '#E0F2FE' }]}>
              <MaterialIcons name="bolt" size={20} color={Colors.softBlue} />
            </View>
            <Text style={styles.featureTitle}>Predictive</Text>
            <Text style={styles.featureDesc}>Identify patterns before they become noticeable.</Text>
          </View>
        </ScrollView>

        <View style={styles.syncContainer}>
          <View style={styles.bluetoothCircleWrapper}>
            <View style={styles.bluetoothCircle}>
              <MaterialIcons name="bluetooth" size={32} color={Colors.violet} />
            </View>
          </View>
          <Text style={styles.syncTitle}>Ready to Sync</Text>
          <Text style={styles.syncDesc}>Seren works best with real-time watch data.</Text>
        </View>

        {watchConnected ? (
          <View style={styles.watchConnectedRow}>
            <MaterialIcons name="check-circle" size={24} color={Colors.sageGreen} />
            <View>
              <Text style={styles.watchName}>Apple Watch Series 9</Text>
              <Text style={styles.watchSub}>Battery 73% · Ready</Text>
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

        <View style={styles.encryptionBox}>
          <View style={styles.encryptionIconWrapper}>
            <MaterialIcons name="lock-outline" size={20} color={Colors.softBlue} />
          </View>
          <View style={styles.encryptionTextBox}>
            <Text style={styles.encryptionTitle}>Local Encryption</Text>
            <Text style={styles.encryptionDesc}>"Keep all data on-device" is active. Your metrics never leave this phone.</Text>
          </View>
          <MaterialIcons name="check-box" size={20} color={Colors.sageGreen} />
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
                <Text style={styles.permDesc}>Used to measure physiological stress.</Text>
              </View>
              <Text style={styles.grantedText}>Granted</Text>
            </View>

            <View style={styles.permissionCard}>
              <View style={styles.permIconWrapperBlue}>
                <MaterialIcons name="bedtime" size={20} color={Colors.softBlue} />
              </View>
              <View style={styles.permTextContainer}>
                <Text style={styles.permLabel}>Sleep Analysis</Text>
                <Text style={styles.permDesc}>Monitors recovery and neural resting.</Text>
              </View>
              <View style={styles.onDeviceBadge}>
                <MaterialIcons name="memory" size={10} color={Colors.sageGreenDark} />
                <Text style={styles.onDeviceBadgeText}>ON-DEVICE PROCESSING</Text>
              </View>
            </View>
          </View>
        </View>

        <TouchableOpacity
          style={styles.startButton}
          onPress={handleGetStarted}
          disabled={!watchConnected}
        >
          <Text style={[styles.startButtonText, watchConnected && { color: Colors.textPrimary }]}>
            Start Your Journey
          </Text>
          <MaterialIcons name="chevron-right" size={20} color={watchConnected ? Colors.textPrimary : Colors.textMuted} />
        </TouchableOpacity>

        <Text style={styles.footerText}>PRIVACY-FIRST MENTAL WELLNESS • V1.0.4</Text>
      </ScrollView>
    );
  }

  return (
    <View style={styles.container}>
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
          <Text style={styles.nextText}>{isLast ? 'Get Started' : 'Continue'}</Text>
          <MaterialIcons name="arrow-forward" size={18} color={Colors.warmWhite} />
        </TouchableOpacity>

        {currentSlide < slides.length - 1 ? (
          <TouchableOpacity onPress={() => setCurrentSlide(slides.length)} style={styles.skipBtn}>
            <Text style={styles.skipText}>Skip setup</Text>
          </TouchableOpacity>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#252240',
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
  // Setup screen
  setupContainer: {
    flex: 1,
    backgroundColor: Colors.cream,
  },
  setupHeader: {
    marginBottom: Spacing.md,
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
  localBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderColor: Colors.sageGreenLight,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Radius.full,
    backgroundColor: Colors.warmWhite,
  },
  localText: {
    fontSize: FontSize.xs,
    color: Colors.sageGreenDark,
    fontWeight: '700',
    letterSpacing: 0.5,
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
  setupHeading: {
    fontSize: FontSize.xxl,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  setupSubtitle: {
    fontSize: FontSize.md,
    color: Colors.textSecondary,
    lineHeight: 24,
  },
  featureCardsContainer: {
    paddingHorizontal: Spacing.xl,
    gap: Spacing.md,
    marginBottom: Spacing.xl,
  },
  featureCard: {
    width: 200,
    padding: Spacing.lg,
    borderRadius: Radius.lg,
    gap: Spacing.md,
  },
  featureIconBg: {
    backgroundColor: Colors.warmWhite,
    width: 40,
    height: 40,
    borderRadius: Radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureTitle: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
  },
  featureDesc: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  syncContainer: {
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  bluetoothCircleWrapper: {
    backgroundColor: Colors.warmWhite,
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.md,
    shadowColor: Colors.textPrimary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.05,
    shadowRadius: 20,
    elevation: 4,
  },
  bluetoothCircle: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.violetMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  syncTitle: {
    fontSize: FontSize.lg,
    fontWeight: FontWeight.bold,
    color: Colors.textPrimary,
    marginBottom: Spacing.xs,
  },
  syncDesc: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  scanButton: {
    flexDirection: 'row',
    backgroundColor: Colors.sageGreen,
    paddingVertical: Spacing.md + 4,
    marginHorizontal: Spacing.xl,
    borderRadius: Radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.md,
    shadowColor: Colors.sageGreen,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  scanButtonActive: {
    backgroundColor: Colors.sageGreenDark,
  },
  scanButtonText: {
    fontSize: FontSize.md,
    fontWeight: FontWeight.bold,
    color: Colors.warmWhite,
  },
  encryptionBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    backgroundColor: '#F0F9FF',
    marginHorizontal: Spacing.xl,
    padding: Spacing.md,
    borderRadius: Radius.lg,
    marginBottom: Spacing.xl,
  },
  encryptionIconWrapper: {
    width: 40,
    height: 40,
    borderRadius: Radius.full,
    backgroundColor: Colors.warmWhite,
    alignItems: 'center',
    justifyContent: 'center',
  },
  encryptionTextBox: {
    flex: 1,
  },
  encryptionTitle: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.bold,
    color: '#0369A1',
    marginBottom: 2,
  },
  encryptionDesc: {
    fontSize: FontSize.xs,
    color: '#38BDF8',
    lineHeight: 16,
  },
  permissionsContainer: {
    marginHorizontal: Spacing.xl,
    marginBottom: Spacing.xl,
  },
  permissionsHeader: {
    fontSize: FontSize.sm,
    fontWeight: 'bold',
    color: Colors.textPrimary,
    letterSpacing: 0.5,
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
  grantedText: {
    fontSize: FontSize.sm,
    fontWeight: 'bold',
    color: Colors.textPrimary,
  },
  onDeviceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#DCFCE7',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    gap: 2,
  },
  onDeviceBadgeText: {
    fontSize: 9,
    fontWeight: 'bold',
    color: Colors.sageGreenDark,
    letterSpacing: 0.5,
  },
  watchConnectedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.md,
  },
  watchName: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  watchSub: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
  },
  startButton: {
    flexDirection: 'row',
    backgroundColor: '#F8FAFC',
    marginHorizontal: Spacing.xl,
    paddingVertical: Spacing.md,
    borderRadius: Radius.lg,
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    marginBottom: Spacing.md,
  },
  startButtonText: {
    fontSize: FontSize.md,
    fontWeight: 'bold',
    color: Colors.textMuted,
  },
  footerText: {
    textAlign: 'center',
    fontSize: 10,
    fontWeight: 'bold',
    color: Colors.textMuted,
    letterSpacing: 1,
    marginTop: Spacing.lg,
  },
});
