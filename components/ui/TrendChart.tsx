import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import Svg, { Path, Circle, Defs, LinearGradient, Stop } from 'react-native-svg';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';

const { width } = Dimensions.get('window');

interface DataPoint {
  label: string;
  value: number;
}

interface TrendChartProps {
  data: DataPoint[];
  color: string;
  height?: number;
  showDots?: boolean;
  showArea?: boolean;
  maxValue?: number;
}

export function TrendChart({ data, color, height = 100, showDots = true, showArea = true, maxValue = 100 }: TrendChartProps) {
  if (!data.length) return null;

  const chartWidth = width - 80;
  const chartHeight = height;
  const points = data.map((d, i) => ({
    x: (i / (data.length - 1)) * chartWidth,
    y: chartHeight - (d.value / maxValue) * chartHeight,
    label: d.label,
    value: d.value,
  }));

  let pathD = `M ${points[0]?.x || 0} ${points[0]?.y || 0}`;
  for (let i = 0; i < points.length - 1; i++) {
    const curr = points[i];
    const next = points[i + 1];
    const midX = (curr.x + next.x) / 2;
    pathD += ` C ${midX} ${curr.y}, ${midX} ${next.y}, ${next.x} ${next.y}`;
  }

  const areaD = `${pathD} L ${points[points.length - 1].x} ${chartHeight} L 0 ${chartHeight} Z`;
  const gradientId = `fade-${color.replace('#', '')}`;

  return (
    <View style={[styles.container, { height: chartHeight + 24 }]}>
      <View style={[styles.chart, { width: chartWidth, height: chartHeight }]}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((fraction, i) => (
          <View
            key={i}
            style={[
              styles.gridLine,
              { top: fraction * chartHeight, width: chartWidth },
            ]}
          />
        ))}

        <Svg width="100%" height="100%" style={StyleSheet.absoluteFill}>
          <Defs>
            <LinearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0" stopColor={color} stopOpacity="0.15" />
              <Stop offset="1" stopColor={color} stopOpacity="0.0" />
            </LinearGradient>
          </Defs>

          {/* Area Fill */}
          {showArea && <Path d={areaD} fill={`url(#${gradientId})`} />}

          {/* Smooth Stroke */}
          <Path d={pathD} fill="none" stroke={color} strokeWidth="2.5" />

          {/* Dots */}
          {showDots && points.map((p, i) => (
            <Circle
              key={`dot-${i}`}
              cx={p.x}
              cy={p.y}
              r={3.5}
              fill={color}
              stroke={Colors.warmWhite}
              strokeWidth={2}
            />
          ))}
        </Svg>
      </View>
    </View>
  );
}

interface BarChartProps {
  data: { label: string; value: number; color?: string }[];
  color: string;
  height?: number;
  maxValue?: number;
}

export function BarChart({ data, color, height = 80, maxValue = 100 }: BarChartProps) {
  return (
    <View style={styles.barContainer}>
      {data.map((d, i) => (
        <View key={i} style={styles.barGroup}>
          <View style={[styles.barTrack, { height }]}>
            <View
              style={[
                styles.bar,
                {
                  height: (d.value / maxValue) * height,
                  backgroundColor: d.color || color,
                },
              ]}
            />
          </View>
          <Text style={styles.barLabel}>{d.label}</Text>
          <Text style={[styles.barValue, { color: d.color || color }]}>{d.value}</Text>
        </View>
      ))}
    </View>
  );
}

export function SleepHeatmap({ data }: { data: number[][] }) {
  const getColor = (val: number) => {
    if (val >= 85) return '#60A5FA';
    if (val >= 70) return '#93C5FD';
    if (val >= 50) return '#DBEAFE';
    return '#F1F5F9';
  };

  return (
    <View style={styles.heatmap}>
      {data.map((week, wi) => (
        <View key={wi} style={styles.heatmapRow}>
          {week.map((val, di) => (
            <View
              key={di}
              style={[
                styles.heatmapCell,
                { backgroundColor: getColor(val) },
              ]}
            />
          ))}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
  chart: {
    position: 'relative',
    overflow: 'hidden',
  },
  gridLine: {
    position: 'absolute',
    height: 1,
    backgroundColor: Colors.warmGray200,
  },
  lineSegment: {
    position: 'absolute',
    height: 2.5,
    borderRadius: 2,
    transformOrigin: 'left center',
  },
  dot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    borderWidth: 2,
  },
  labelsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 6,
  },
  label: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  barContainer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 6,
  },
  barGroup: {
    flex: 1,
    alignItems: 'center',
    gap: 4,
  },
  barTrack: {
    width: '100%',
    backgroundColor: Colors.warmGray200,
    borderRadius: Radius.sm,
    justifyContent: 'flex-end',
  },
  bar: {
    borderRadius: Radius.sm,
    width: '100%',
  },
  barLabel: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontWeight: FontWeight.medium,
  },
  barValue: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
  },
  heatmap: {
    gap: Spacing.sm,
    width: '100%',
  },
  heatmapRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  heatmapCell: {
    width: 32,
    height: 32,
    borderRadius: 16,
  },
});
