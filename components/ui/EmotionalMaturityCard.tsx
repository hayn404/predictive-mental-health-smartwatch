/**
 * Seren — "How well is Seren reading me?" card
 * ===============================================
 * User-facing view of services/ai/emotionalMaturity.ts's MaturitySnapshot.
 *
 * Deliberately does NOT show the snapshot's raw `evidence`/`nextStep` text —
 * those are written for developers (file names, "ALTER TABLE", etc., see
 * docs/MATURITY_CHECKLIST.md) and would read as cold/clinical here. This
 * component maps the same dimension ids to short, warm, plain-language
 * copy instead, so both audiences read the same underlying numbers.
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, LayoutAnimation, Platform, UIManager } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from './GlassCard';
import { MaturityDimension, MaturitySnapshot } from '@/services/ai/emotionalMaturity';

if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

interface EmotionalMaturityCardProps {
  snapshot: MaturitySnapshot;
}

// Warm, plain-language copy per dimension — intentionally separate from
// emotionalMaturity.ts's dev-facing `evidence`/`nextStep` strings.
const FRIENDLY_COPY: Record<string, { label: string; description: string }> = {
  voice_capture_quality: {
    label: 'Hearing you clearly',
    description: 'How reliably your voice check-ins are being captured.',
  },
  understanding_depth: {
    label: 'Understanding depth',
    description: 'How deeply your words are being analyzed, beyond surface keywords.',
  },
  emotional_vocabulary_breadth: {
    label: 'Range of emotions noticed',
    description: 'How many different feelings Seren has learned to recognize in you.',
  },
  body_mind_cross_referencing: {
    label: 'Connecting body & words',
    description: 'How often Seren checks what you say against what your body shows.',
  },
  conversational_responsiveness: {
    label: 'Asking the right questions',
    description: 'How often Seren follows up with a thoughtful question.',
  },
  duet_latency: {
    label: 'Conversation flow',
    description: 'How natural the back-and-forth feels when you talk with Seren.',
  },
  trend_stability: {
    label: 'Getting to know your patterns',
    description: 'How much history Seren has to spot real patterns vs. one-off moments.',
  },
};

const TIER_DESCRIPTIONS: Record<string, string> = {
  'Just getting to know you': "We're just starting out — check in a bit more and Seren will pick up on your patterns.",
  'Learning your patterns': 'Seren is starting to notice patterns in how you check in.',
  'Reading you well': 'Seren has a solid read on your emotional patterns by now.',
  'Deeply attuned': "Seren's understanding of you is quite deep at this point.",
  'Fully attuned': "Seren is reading you about as well as it currently can.",
};

export function EmotionalMaturityCard({ snapshot }: EmotionalMaturityCardProps) {
  const [expanded, setExpanded] = useState(false);

  const toggle = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpanded(v => !v);
  };

  return (
    <GlassCard variant="violet" style={styles.card}>
      <TouchableOpacity onPress={toggle} activeOpacity={0.8} style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.iconBg}>
            <MaterialIcons name="psychology" size={18} color={Colors.violet} />
          </View>
          <View style={styles.headerText}>
            <Text style={styles.title}>How well Seren is reading you</Text>
            <Text style={styles.tier}>{snapshot.overallTier}</Text>
          </View>
        </View>
        <MaterialIcons
          name={expanded ? 'expand-less' : 'expand-more'}
          size={22}
          color={Colors.textMuted}
        />
      </TouchableOpacity>

      <View style={styles.scoreRow}>
        <View style={styles.scoreTrack}>
          <View style={[styles.scoreFill, { width: `${Math.max(4, snapshot.score)}%` }]} />
        </View>
        <Text style={styles.scoreLabel}>{snapshot.score}%</Text>
      </View>

      <Text style={styles.tierDescription}>
        {TIER_DESCRIPTIONS[snapshot.overallTier] ?? ''}
      </Text>

      {expanded && (
        <View style={styles.dimensionList}>
          {snapshot.dimensions.map(d => (
            <DimensionRow key={d.id} dimension={d} />
          ))}
          {snapshot.sampleSize === 0 && (
            <Text style={styles.emptyHint}>Complete a check-in to start building this picture.</Text>
          )}
        </View>
      )}
    </GlassCard>
  );
}

function DimensionRow({ dimension }: { dimension: MaturityDimension }) {
  const copy = FRIENDLY_COPY[dimension.id] ?? { label: dimension.label, description: '' };
  return (
    <View style={styles.dimensionRow}>
      <View style={styles.dimensionDots}>
        {[0, 1, 2, 3].map(i => (
          <View
            key={i}
            style={[
              styles.dot,
              i < dimension.level ? styles.dotFilled : styles.dotEmpty,
            ]}
          />
        ))}
      </View>
      <View style={styles.dimensionTextWrap}>
        <Text style={styles.dimensionLabel}>{copy.label}</Text>
        {copy.description ? (
          <Text style={styles.dimensionDescription}>{copy.description}</Text>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    flex: 1,
  },
  iconBg: {
    width: 32,
    height: 32,
    borderRadius: Radius.md,
    backgroundColor: Colors.violetMuted,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerText: {
    flex: 1,
  },
  title: {
    fontSize: FontSize.sm,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  tier: {
    fontSize: FontSize.xs,
    color: Colors.violet,
    fontWeight: FontWeight.medium,
    marginTop: 1,
  },
  scoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  scoreTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.violetMuted,
    overflow: 'hidden',
  },
  scoreFill: {
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.violet,
  },
  scoreLabel: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    color: Colors.violet,
    width: 36,
    textAlign: 'right',
  },
  tierDescription: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    lineHeight: 18,
  },
  dimensionList: {
    marginTop: Spacing.xs,
    gap: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.warmGray200,
    paddingTop: Spacing.sm,
  },
  dimensionRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.sm,
  },
  dimensionDots: {
    flexDirection: 'row',
    gap: 3,
    marginTop: 4,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  dotFilled: {
    backgroundColor: Colors.violet,
  },
  dotEmpty: {
    backgroundColor: Colors.warmGray200,
  },
  dimensionTextWrap: {
    flex: 1,
  },
  dimensionLabel: {
    fontSize: FontSize.xs,
    fontWeight: FontWeight.semibold,
    color: Colors.textPrimary,
  },
  dimensionDescription: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    marginTop: 1,
    lineHeight: 16,
  },
  emptyHint: {
    fontSize: FontSize.xs,
    color: Colors.textMuted,
    fontStyle: 'italic',
  },
});
