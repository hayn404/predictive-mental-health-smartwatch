import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, TextInput, Keyboard, TouchableWithoutFeedback, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors, FontSize, FontWeight, Spacing, Radius } from '@/constants/theme';
import { GlassCard } from '@/components/ui/GlassCard';

export default function JournalingActivityScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    const [entry, setEntry] = useState('');

    const handleSave = () => {
        if (!entry.trim()) {
            Alert.alert("Empty Entry", "Please write something before saving.");
            return;
        }
        Keyboard.dismiss();
        // Here you would typically save to AsyncStorage or a backend
        Alert.alert("Saved", "Your journal entry has been safely stored locally.", [
            { text: "OK", onPress: () => router.back() }
        ]);
    };

    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={[styles.header, { paddingTop: insets.top + Spacing.md }]}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <MaterialIcons name="chevron-left" size={28} color={Colors.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Journaling</Text>
                <TouchableOpacity onPress={() => router.push('/(tabs)')} style={styles.closeBtn}>
                    <MaterialIcons name="close" size={24} color={Colors.textMuted} />
                </TouchableOpacity>
            </View>

            <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
                <View style={styles.content}>
                    <View style={styles.promptContainer}>
                        <MaterialIcons name="edit" size={32} color={Colors.violet} style={{ marginBottom: 12 }} />
                        <Text style={styles.promptText}>What is on your mind right now?</Text>
                        <Text style={styles.promptSub}>Externalize your thoughts to identify patterns.</Text>
                    </View>

                    <GlassCard variant="default" style={styles.inputCard}>
                        <TextInput
                            style={styles.input}
                            placeholder="Start typing..."
                            placeholderTextColor={Colors.warmGray400}
                            multiline
                            textAlignVertical="top"
                            returnKeyType="done"
                            blurOnSubmit={true}
                            onSubmitEditing={Keyboard.dismiss}
                            value={entry}
                            onChangeText={setEntry}
                        />
                    </GlassCard>

                    <View style={styles.footer}>
                        <TouchableOpacity style={[styles.saveBtn, !entry.trim() && styles.saveBtnDisabled]} onPress={handleSave}>
                            <Text style={styles.saveBtnText}>Save Entry</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </TouchableWithoutFeedback>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.warmWhite, // Changed to white
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
    backBtn: { padding: 4 },
    closeBtn: { padding: 4 },
    headerTitle: {
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
    },
    content: {
        flex: 1,
        padding: Spacing.lg,
    },
    promptContainer: {
        alignItems: 'center',
        marginTop: Spacing.xl,
        marginBottom: Spacing.xxl,
    },
    promptText: {
        fontSize: FontSize.xl,
        fontWeight: FontWeight.bold,
        color: Colors.textPrimary,
        marginBottom: Spacing.xs,
        textAlign: 'center',
    },
    promptSub: {
        fontSize: FontSize.sm,
        color: Colors.textSecondary,
        textAlign: 'center',
    },
    inputCard: {
        flex: 1,
        backgroundColor: '#FAFAFA',
        borderWidth: 1,
        borderColor: '#E5E7EB',
        borderRadius: Radius.lg,
        padding: Spacing.md,
    },
    input: {
        flex: 1,
        fontSize: FontSize.md,
        color: Colors.textPrimary,
        lineHeight: 24,
    },
    footer: {
        marginTop: Spacing.xl,
    },
    saveBtn: {
        backgroundColor: Colors.violet,
        width: '100%',
        paddingVertical: Spacing.lg,
        borderRadius: Radius.full,
        alignItems: 'center',
    },
    saveBtnDisabled: {
        backgroundColor: Colors.warmGray300,
    },
    saveBtnText: {
        color: Colors.warmWhite,
        fontSize: FontSize.md,
        fontWeight: FontWeight.bold,
    },
});
