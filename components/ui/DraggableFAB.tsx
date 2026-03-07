import React, { useRef } from 'react';
import { StyleSheet, Animated, PanResponder, TouchableOpacity, useWindowDimensions } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter, usePathname } from 'expo-router';
import { Colors } from '@/constants/theme';

export function DraggableFAB() {
    const router = useRouter();
    const pathname = usePathname();
    const insets = useSafeAreaInsets();
    const { width: screenWidth, height: screenHeight } = useWindowDimensions();

    const FAB_SIZE = 56;

    // Initial position: Bottom right
    const initialPosition = {
        x: screenWidth - FAB_SIZE - 24, // 24px padding from right
        y: screenHeight - insets.bottom - FAB_SIZE - 100, // Above bottom tabs
    };

    const pan = useRef(new Animated.ValueXY(initialPosition)).current;

    const panResponder = useRef(
        PanResponder.create({
            onMoveShouldSetPanResponder: (_, gestureState) => {
                // Only act as a pan responder if the user has dragged a certain distance (distinguish from tap)
                return Math.abs(gestureState.dx) > 5 || Math.abs(gestureState.dy) > 5;
            },
            onPanResponderGrant: () => {
                pan.setOffset({
                    x: (pan.x as any)._value,
                    y: (pan.y as any)._value,
                });
                pan.setValue({ x: 0, y: 0 });
            },
            onPanResponderMove: Animated.event(
                [null, { dx: pan.x, dy: pan.y }],
                { useNativeDriver: false }
            ),
            onPanResponderRelease: () => {
                pan.flattenOffset();

                // Optional: Snap to edges logic
                const currentX = (pan.x as any)._value;
                const currentY = (pan.y as any)._value;

                // Restrict to screen bounds
                let toX = currentX;
                let toY = currentY;

                // Snap to left or right edge
                if (currentX < screenWidth / 2) {
                    toX = 24; // left padding
                } else {
                    toX = screenWidth - FAB_SIZE - 24; // right padding
                }

                // Keep vertically within safe bounds
                const minPosY = insets.top + 24;
                const maxPosY = screenHeight - insets.bottom - FAB_SIZE - 24;

                if (currentY < minPosY) toY = minPosY;
                if (currentY > maxPosY) toY = maxPosY;

                Animated.spring(pan, {
                    toValue: { x: toX, y: toY },
                    useNativeDriver: false, // Layout animations cannot use native driver
                    friction: 5,
                }).start();
            },
        })
    ).current;

    // Hide FAB on onboarding pages ONLY. Wait for router to mount before aggressive hiding.
    if (pathname && pathname.includes('/onboarding')) {
        return null;
    }

    return (
        <Animated.View
            style={[
                styles.fabContainer,
                { transform: [{ translateX: pan.x }, { translateY: pan.y }] }
            ]}
            {...panResponder.panHandlers}
        >
            <TouchableOpacity
                style={styles.fab}
                onPress={() => router.push('/(tabs)/checkin')}
                activeOpacity={0.8}
            >
                <MaterialIcons name="mic" size={24} color={Colors.warmWhite} />
            </TouchableOpacity>
        </Animated.View>
    );
}

const styles = StyleSheet.create({
    fabContainer: {
        position: 'absolute',
        zIndex: 999,
        elevation: 999,
    },
    fab: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: Colors.violet,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 4,
        borderColor: Colors.warmWhite,
        shadowColor: '#1F212D',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 12,
        elevation: 6,
    },
});
