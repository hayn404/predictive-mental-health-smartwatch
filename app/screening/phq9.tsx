import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from '@/components/ui/GlassCard';

const QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way",
];

const OPTIONS = [
    { label: 'Not at all', score: 0 },
    { label: 'Several days', score: 1 },
    { label: 'More than half the days', score: 2 },
    { label: 'Nearly every day', score: 3 },
];

export default function Phq9Screen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [answers, setAnswers] = useState<Record<number, number>>({});
    const [currentStep, setCurrentStep] = useState(0);

    const handleSelect = (score: number) => {
        setAnswers(prev => ({ ...prev, [currentStep]: score }));

        // Auto-advance after a tiny delay for visual feedback
        setTimeout(() => {
            setCurrentStep(prev => prev + 1);
        }, 300);
    };

    const handleBack = () => {
        if (currentStep > 0) {
            setCurrentStep(prev => prev - 1);
        } else {
            router.back();
        }
    };

    const calculateScore = () => {
        return Object.values(answers).reduce((sum, score) => sum + score, 0);
    };

    const getSeverity = (score: number) => {
        if (score <= 4) return { text: 'Minimal', color: Colors.sageGreen };
        if (score <= 9) return { text: 'Mild', color: Colors.softBlue };
        if (score <= 14) return { text: 'Moderate', color: Colors.warning };
        if (score <= 19) return { text: 'Moderately Severe', color: Colors.error };
        return { text: 'Severe', color: Colors.error };
    };

    const isComplete = currentStep >= QUESTIONS.length;
    const progress = isComplete ? 1 : currentStep / QUESTIONS.length;

    return (
        <View style={styles.container}>
            <View style={[styles.header, { paddingTop: insets.top + Spacing.md }]}>
                <TouchableOpacity onPress={handleBack} style={styles.backBtn}>
                    <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>PHQ-9 Assessment</Text>
                <TouchableOpacity onPress={() => router.back()} style={styles.closeBtn}>
                    <MaterialIcons name="close" size={24} color={Colors.textMuted} />
                </TouchableOpacity>
            </View>

            {/* Progress Bar */}
            <View style={styles.progressContainer}>
                <View style={styles.progressTextRow}>
                    <Text style={styles.progressText}>
                        {isComplete ? 'Complete' : `Question ${currentStep + 1} of ${QUESTIONS.length}`}
                    </Text>
                </View>
                <View style={styles.progressBarBg}>
                    <View style={[styles.progressBarFill, { width: `${progress * 100}%` }]} />
                </View>
            </View>

            <View style={styles.content}>
                {!isComplete ? (
                    <View style={styles.stepContainer}>
                        <Text style={styles.subtitle}>
                            Over the last 2 weeks, how often have you been bothered by:
                        </Text>

                        <GlassCard variant="default" style={styles.questionCard}>
                            <Text style={styles.questionText}>&quot;{QUESTIONS[currentStep]}&quot;</Text>
                        </GlassCard>

                        <View style={styles.optionsContainer}>
                            {OPTIONS.map((opt) => {
                                const isSelected = answers[currentStep] === opt.score;
                                return (
                                    <TouchableOpacity
                                        key={opt.score}
                                        style={[
                                            styles.optionBtn,
                                            isSelected && styles.optionBtnSelected
                                        ]}
                                        onPress={() => handleSelect(opt.score)}
                                        activeOpacity={0.7}
                                    >
                                        <View style={[styles.radioOuter, isSelected && styles.radioOuterSelected]}>
                                            {isSelected && <View style={styles.radioInner} />}
                                        </View>
                                        <Text style={[styles.optionText, isSelected && styles.optionTextSelected]}>
                                            {opt.label}
                                        </Text>
                                    </TouchableOpacity>
                                );
                            })}
                        </View>
                    </View>
                ) : (
                    <View style={styles.resultContainer}>
                        <GlassCard variant="default" style={styles.resultCard}>
                            <MaterialIcons name="verified" size={48} color={Colors.sageGreen} style={{ marginBottom: 8 }} />
                            <Text style={styles.resultTitle}>Assessment Complete</Text>
                            <Text style={styles.resultScore}>{calculateScore()} / 27</Text>

                            <View style={[styles.severityBadge, { backgroundColor: getSeverity(calculateScore()).color + '20' }]}>
                                <Text style={[styles.severityText, { color: getSeverity(calculateScore()).color }]}>
                                    {getSeverity(calculateScore()).text} Depression
                                </Text>
                            </View>

                            <View style={styles.divider} />

                            <Text style={styles.disclaimer}>
                                This is a standard screening tool, not a clinical diagnosis. If you are experiencing distress, please consult a healthcare professional.
                            </Text>

                            <TouchableOpacity style={styles.doneBtn} onPress={() => router.back()}>
                                <Text style={styles.doneBtnText}>Return Home</Text>
                            </TouchableOpacity>
                        </GlassCard>
                    </View>
                )}
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
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: Spacing.md,
        paddingBottom: Spacing.md,
        backgroundColor: Colors.warmWhite,
        borderBottomWidth: 1,
        borderBottomColor: Colors.warmGray200,
    },
    backBtn: {
        padding: 4,
    },
    closeBtn: {
        padding: 4,
    },
    headerTitle: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    progressContainer: {
        paddingHorizontal: Spacing.xl,
        paddingTop: Spacing.lg,
        paddingBottom: Spacing.sm,
    },
    progressTextRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    progressText: {
        fontSize: FontSize.xs,
        fontWeight: 'bold',
        color: Colors.textMuted,
        letterSpacing: 0.5,
        textTransform: 'uppercase',
    },
    progressBarBg: {
        height: 6,
        backgroundColor: Colors.warmGray200,
        borderRadius: Radius.full,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        backgroundColor: Colors.softBlue,
        borderRadius: Radius.full,
    },
    content: {
        flex: 1,
        padding: Spacing.lg,
    },
    stepContainer: {
        flex: 1,
    },
    subtitle: {
        fontSize: FontSize.md,
        fontWeight: '500',
        color: Colors.textSecondary,
        lineHeight: 24,
        marginBottom: Spacing.lg,
        textAlign: 'center',
    },
    questionCard: {
        padding: Spacing.xl,
        backgroundColor: Colors.warmWhite,
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 140,
        marginBottom: Spacing.xl,
        borderWidth: 1.5,
        borderColor: '#E0F2FE',
    },
    questionText: {
        fontSize: FontSize.lg,
        fontWeight: FontWeight.semibold,
        color: Colors.softBlueDark,
        textAlign: 'center',
        lineHeight: 28,
    },
    optionsContainer: {
        gap: Spacing.md,
    },
    optionBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: Spacing.lg,
        backgroundColor: Colors.warmWhite,
        borderRadius: Radius.xl,
        borderWidth: 1,
        borderColor: Colors.warmGray200,
    },
    optionBtnSelected: {
        backgroundColor: '#F0F9FF',
        borderColor: Colors.softBlue,
    },
    radioOuter: {
        width: 22,
        height: 22,
        borderRadius: 11,
        borderWidth: 2,
        borderColor: Colors.warmGray400,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.md,
    },
    radioOuterSelected: {
        borderColor: Colors.softBlue,
    },
    radioInner: {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: Colors.softBlue,
    },
    optionText: {
        fontSize: FontSize.md,
        color: Colors.textSecondary,
    },
    optionTextSelected: {
        color: Colors.softBlueDark,
        fontWeight: FontWeight.bold,
    },
    resultContainer: {
        flex: 1,
        justifyContent: 'center',
    },
    resultCard: {
        padding: Spacing.xl,
        alignItems: 'center',
        gap: Spacing.md,
        backgroundColor: Colors.warmWhite,
    },
    resultTitle: {
        fontSize: FontSize.xl,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    resultScore: {
        fontSize: 56,
        fontWeight: FontWeight.bold,
        color: Colors.softBlue,
    },
    severityBadge: {
        paddingHorizontal: Spacing.xl,
        paddingVertical: Spacing.sm,
        borderRadius: Radius.full,
        marginBottom: Spacing.sm,
    },
    severityText: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
    },
    divider: {
        height: 1,
        width: '100%',
        backgroundColor: Colors.warmGray200,
        marginVertical: Spacing.sm,
    },
    disclaimer: {
        fontSize: FontSize.xs,
        color: Colors.textMuted,
        textAlign: 'center',
        lineHeight: 18,
        marginBottom: Spacing.lg,
    },
    doneBtn: {
        backgroundColor: Colors.softBlue,
        width: '100%',
        paddingVertical: Spacing.lg,
        borderRadius: Radius.full,
        alignItems: 'center',
    },
    doneBtnText: {
        color: Colors.warmWhite,
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
    },
});
