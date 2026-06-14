import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import Svg, { Rect, Line, Text as SvgText } from 'react-native-svg';
import { Colors, FontSize, FontWeight, Spacing } from '@/constants/theme';
import type { RawSleepStage, SleepStageType } from '@/services/ai/types';

const { width } = Dimensions.get('window');

// Hypnogram rows, top -> bottom (clinical convention: Awake on top, Deep at the bottom).
const ROWS = ['awake', 'rem', 'light', 'deep'] as const;
type Row = (typeof ROWS)[number];

const LABELS: Record<Row, string> = { awake: 'Awake', rem: 'REM', light: 'Light', deep: 'Deep' };
const ROW_COLOR: Record<Row, string> = {
  awake: '#E8A87C', // amber
  rem: '#9B8EC4',   // lavender
  light: '#60A5FA', // blue
  deep: '#3B5BA5',  // deep blue
};

// Map every SleepStageType onto one of the 4 hypnogram rows.
function rowOf(stage: SleepStageType): Row {
  switch (stage) {
    case 'awake':
    case 'out_of_bed':
      return 'awake';
    case 'rem':
      return 'rem';
    case 'deep':
      return 'deep';
    default:
      return 'light'; // light, sleeping, unknown
  }
}

interface HypnogramProps {
  stages: RawSleepStage[] | undefined;
  /** Mean per-epoch ML confidence (0-1). Shown only when present (on-device model). */
  confidence?: number;
  height?: number;
  cardHorizontalPadding?: number;
}

export function Hypnogram({ stages, confidence, height = 116, cardHorizontalPadding = 80 }: HypnogramProps) {
  if (!stages || stages.length === 0) {
    return <Text style={styles.empty}>No stage timeline for last night.</Text>;
  }

  const svgW = width - cardHorizontalPadding;
  const labelW = 42;
  const chartW = Math.max(40, svgW - labelW - 6);
  const topPad = 6;
  const rowH = (height - topPad * 2) / ROWS.length;

  const t0 = stages[0].startTime;
  const t1 = stages[stages.length - 1].endTime;
  const span = Math.max(1, t1 - t0);

  const xAt = (t: number) => labelW + ((t - t0) / span) * chartW;
  const rowCenterY = (r: Row) => topPad + ROWS.indexOf(r) * rowH + rowH / 2;

  const fmt = (t: number) =>
    new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

  return (
    <View>
      <Svg width={svgW} height={height + 16}>
        {ROWS.map((r, i) => {
          const y = topPad + i * rowH + rowH / 2;
          return (
            <React.Fragment key={r}>
              <Line x1={labelW} y1={y} x2={labelW + chartW} y2={y} stroke="#EEF2F7" strokeWidth={1} />
              <SvgText x={0} y={y + 3} fontSize={9} fill={Colors.textMuted ?? '#94A3B8'}>
                {LABELS[r]}
              </SvgText>
            </React.Fragment>
          );
        })}

        {stages.map((seg, i) => {
          const r = rowOf(seg.stage);
          const xs = xAt(seg.startTime);
          const xe = Math.max(xs + 2, xAt(seg.endTime));
          return (
            <Rect
              key={i}
              x={xs}
              y={rowCenterY(r) - 4}
              width={xe - xs}
              height={8}
              rx={3}
              fill={ROW_COLOR[r]}
            />
          );
        })}

        {/* start / end time ticks */}
        <SvgText x={labelW} y={height + 10} fontSize={9} fill={Colors.textMuted ?? '#94A3B8'}>
          {fmt(t0)}
        </SvgText>
        <SvgText x={labelW + chartW} y={height + 10} fontSize={9} fill={Colors.textMuted ?? '#94A3B8'} textAnchor="end">
          {fmt(t1)}
        </SvgText>
      </Svg>

      {typeof confidence === 'number' && (
        <View style={styles.confidenceRow}>
          <View style={styles.mlDot} />
          <Text style={styles.confidenceText}>
            On-device model · {Math.round(confidence * 100)}% mean confidence
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  empty: {
    fontSize: FontSize.sm,
    color: Colors.textMuted ?? '#94A3B8',
    paddingVertical: Spacing.md,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.xs,
  },
  mlDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#9B8EC4',
    marginRight: 6,
  },
  confidenceText: {
    fontSize: FontSize.xs ?? 11,
    fontWeight: FontWeight.medium as any,
    color: Colors.textSecondary ?? '#64748B',
  },
});
