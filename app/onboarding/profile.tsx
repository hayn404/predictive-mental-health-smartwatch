import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    TextInput,
    KeyboardAvoidingView,
    Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { Image } from 'expo-image';
import { useWellness } from '@/hooks/useWellness';

export default function ProfileScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const { setChronologicalAge } = useWellness();
    const [age, setAge] = useState('');
    const [error, setError] = useState('');

    const handleNext = () => {
        const a = parseInt(age, 10);
        if (isNaN(a) || a < 5 || a > 120) {
            setError('Please enter a valid age (5–120).');
            return;
        }
        setChronologicalAge(a);            // persisted on-device (chronological_age)
        router.push('/onboarding/sync');
    };

    return (
        <View style={styles.container}>
            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : undefined}
                style={{ flex: 1 }}
            >
                <ScrollView
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={{ paddingTop: insets.top + Spacing.xl, paddingBottom: insets.bottom + Spacing.xl }}
                >
                    <View style={styles.header}>
                        <View style={styles.titleRow}>
                            <View style={styles.appIconContainer}>
                                <Image source={require('@/assets/images/seren-brain.png')} style={styles.appIcon} />
                            </View>
                            <Text style={styles.setupTitle}>About You</Text>
                        </View>

                        <View style={styles.progressRow}>
                            <Text style={styles.stepText}>STEP 1 OF 3</Text>
                            <Text style={styles.stepSubText}>Your Profile</Text>
                        </View>
                        <View style={styles.progressBarBg}>
                            <View style={[styles.progressBarFill, { width: '33%' }]} />
                        </View>

                        <Text style={styles.heading}>A little about you.</Text>
                        <Text style={styles.subtitle}>
                            Your age personalizes your physiological &quot;heart age&quot; and the
                            age-gap insight. It stays on your device.
                        </Text>
                    </View>

                    <View style={styles.field}>
                        <Text style={styles.label}>Your age</Text>
                        <TextInput
                            style={styles.input}
                            value={age}
                            onChangeText={(t) => { setAge(t.replace(/[^0-9]/g, '')); setError(''); }}
                            keyboardType="number-pad"
                            placeholder="e.g. 25"
                            placeholderTextColor={Colors.textMuted}
                            maxLength={3}
                            returnKeyType="done"
                            onSubmitEditing={handleNext}
                        />
                        {error ? <Text style={styles.error}>{error}</Text> : null}
                    </View>

                    <TouchableOpacity style={styles.startButton} onPress={handleNext}>
                        <Text style={styles.startButtonText}>Continue</Text>
                        <MaterialIcons name="arrow-forward" size={20} color={Colors.warmWhite} />
                    </TouchableOpacity>
                </ScrollView>
            </KeyboardAvoidingView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.cream },
    header: { marginBottom: Spacing.xl, paddingHorizontal: Spacing.xl },
    titleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: Spacing.lg },
    appIconContainer: { width: 32, height: 32, borderRadius: Radius.md, overflow: 'hidden', backgroundColor: Colors.warmWhite },
    appIcon: { width: '100%', height: '100%', resizeMode: 'contain' },
    setupTitle: { fontSize: FontSize.lg, fontWeight: FontWeight.bold, color: Colors.textPrimary },
    progressRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: Spacing.sm },
    stepText: { fontSize: FontSize.xs, color: Colors.violet, fontWeight: '700', letterSpacing: 1.2 },
    stepSubText: { fontSize: FontSize.xs, color: Colors.textSecondary, fontWeight: '500' },
    progressBarBg: { height: 4, backgroundColor: Colors.violetMuted, borderRadius: Radius.full, marginBottom: Spacing.xl },
    progressBarFill: { height: 4, backgroundColor: Colors.violet, borderRadius: Radius.full },
    heading: { fontSize: FontSize.xxl, fontWeight: FontWeight.bold, color: Colors.textPrimary, marginBottom: Spacing.xs },
    subtitle: { fontSize: FontSize.md, color: Colors.textSecondary, lineHeight: 24 },
    field: { paddingHorizontal: Spacing.xl, marginBottom: Spacing.xl },
    label: { fontSize: FontSize.sm, fontWeight: FontWeight.semibold, color: Colors.textPrimary, marginBottom: Spacing.sm },
    input: {
        backgroundColor: Colors.warmWhite,
        borderRadius: Radius.md,
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.md,
        fontSize: FontSize.lg,
        color: Colors.textPrimary,
        borderWidth: 1,
        borderColor: Colors.warmGray300,
    },
    error: { color: Colors.error, fontSize: FontSize.sm, marginTop: Spacing.sm },
    startButton: {
        flexDirection: 'row',
        backgroundColor: Colors.sageGreen,
        paddingVertical: Spacing.lg,
        marginHorizontal: Spacing.xl,
        borderRadius: Radius.md,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
    },
    startButtonText: { fontSize: FontSize.md, fontWeight: FontWeight.bold, color: Colors.warmWhite },
});
