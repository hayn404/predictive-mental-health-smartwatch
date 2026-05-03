import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Easing,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';

interface PanicInterventionBannerProps {
  onClose: () => void;
  onStartBreathing: () => void;
  autoDismissMs?: number;
}

export function PanicInterventionBanner({
  onClose,
  onStartBreathing,
  autoDismissMs = 10000,
}: PanicInterventionBannerProps) {
  const slideAnim = useRef(new Animated.Value(-100)).current;

  useEffect(() => {
    // Slide in animation
    Animated.timing(slideAnim, {
      toValue: 0,
      duration: 200,
      easing: Easing.out(Easing.ease),
      useNativeDriver: true,
    }).start();

    // Auto-dismiss timer
    const timer = setTimeout(() => {
      handleClose();
    }, autoDismissMs);

    return () => clearTimeout(timer);
  }, []);

  const handleClose = () => {
    Animated.timing(slideAnim, {
      toValue: -100,
      duration: 200,
      easing: Easing.in(Easing.ease),
      useNativeDriver: true,
    }).start(() => {
      onClose();
    });
  };

  return (
    <Animated.View
      style={[
        styles.container,
        {
          transform: [{ translateY: slideAnim }],
        },
      ]}
    >
      <View style={styles.bannerContent}>
        <View style={styles.textContent}>
          <MaterialIcons name="warning" size={20} color="#FFFFFF" />
          <Text style={styles.bannerText}>Elevated anxiety detected. Start a breathing exercise?</Text>
        </View>
        <TouchableOpacity
          onPress={handleClose}
          hitSlop={10}
          activeOpacity={0.7}
        >
          <MaterialIcons name="close" size={20} color="#FFFFFF" />
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        onPress={() => {
          handleClose();
          setTimeout(onStartBreathing, 200);
        }}
        style={styles.ctaButton}
        activeOpacity={0.8}
      >
        <Text style={styles.ctaText}>Box Breathing</Text>
        <MaterialIcons name="arrow-forward" size={16} color="#FFFFFF" style={{ marginLeft: 6 }} />
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#FF4D4D',
    borderRadius: Radius.lg,
    marginHorizontal: Spacing.lg,
    marginVertical: Spacing.sm,
    overflow: 'hidden',
    shadowColor: '#FF4D4D',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  bannerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: Spacing.md,
    gap: Spacing.md,
  },
  textContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  bannerText: {
    flex: 1,
    fontSize: FontSize.sm,
    fontWeight: '600',
    color: '#FFFFFF',
    lineHeight: 18,
  },
  ctaButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    marginHorizontal: Spacing.md,
    marginBottom: Spacing.md,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  ctaText: {
    fontSize: FontSize.sm,
    fontWeight: 'bold',
    color: '#FFFFFF',
    letterSpacing: 0.3,
  },
});
