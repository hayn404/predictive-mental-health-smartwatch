import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';
import { FontSize } from '@/constants/theme';
import { FocusLevel } from '@/services/ai/types';
import { getFocusColor } from '@/services/ai/focusModel';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface FocusGaugeProps {
  value: number;       // 0-100
  level: FocusLevel;
  size?: number;
}


export function FocusGauge({ value, level, size = 200 }: FocusGaugeProps) {
  const color = getFocusColor(level);
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: value,
      duration: 1200,
      useNativeDriver: true,
    }).start();
  }, [value]);

  const strokeWidth = 14;
  const radius = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;

  const strokeDashoffset = animatedValue.interpolate({
    inputRange: [0, 100],
    outputRange: [circumference, 0],
  });

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <G rotation="-90" origin={`${size / 2}, ${size / 2}`}>
          {/* Track ring */}
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#F0F0F0"
            strokeWidth={strokeWidth}
            fill="none"
          />
          {/* Filled arc */}
          <AnimatedCircle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="none"
          />
        </G>
      </Svg>

      {/* Center content */}
      <View style={styles.center}>
        <Text style={[styles.score, { color }]}>{Math.round(value)}</Text>
        <Text style={styles.outOf}>/100</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  center: {
    position: 'absolute',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },
  score: {
    fontSize: 42,
    fontWeight: '800',
    lineHeight: 48,
  },
  outOf: {
    fontSize: FontSize.xs,
    fontWeight: '600',
    color: '#9CA3AF',
    marginTop: -4,
  },
});
