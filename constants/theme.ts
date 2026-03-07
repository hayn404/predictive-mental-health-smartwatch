// Seren Mental Health App - Design Tokens
// Color palette matched to UI design: lavender background, purple primary, mint green accent
export const Colors = {
  // Brand Palette — Primary Purple/Violet (from design)
  sageGreen: '#35e27e',          // Mint green accent (tab active, badges, HR line)
  sageGreenLight: '#85E895',
  sageGreenDark: '#32A04A',
  sageGreenMuted: '#E8F9EA',

  softBlue: '#739EE8',           // Steel blue (sleep dots)
  softBlueLight: '#A8C5F5',
  softBlueDark: '#4A71B5',
  softBlueMuted: '#EEF4FD',

  violet: '#A288FC',             // Primary purple (buttons, waveform, stress line)
  violetLight: '#C3B3FD',
  violetDark: '#7054D6',
  violetMuted: '#F2EEFF',

  // Background palette
  cream: '#f8fafa',              // App background (bright, clean white-gray)
  warmWhite: '#FFFFFF',          // Card background
  warmGray100: '#F0F1F5',        // Subtle dividers/inputs
  warmGray200: '#E4E5EB',
  warmGray300: '#D1D3DC',
  warmGray400: '#A1A3B1',
  warmGray500: '#7B7E8F',
  warmGray600: '#5A5D6E',
  warmGray700: '#3D404F',
  warmGray800: '#1F212D',

  // Semantic
  success: '#4CD964',
  warning: '#FFB84D',
  warningMuted: '#FFF4E5',
  error: '#FF4D4D',
  errorMuted: '#FFE5E5',
  info: '#739EE8',

  // Surface / Glass
  glass: 'rgba(255, 255, 255, 0.90)',
  glassBorder: 'rgba(162, 136, 252, 0.12)',
  glassDeep: 'rgba(255, 255, 255, 0.60)',
  overlay: 'rgba(31, 33, 45, 0.45)',

  // Text
  textPrimary: '#1F212D',
  textSecondary: '#5A5D6E',
  textMuted: '#A1A3B1',
  textOnDark: '#FFFFFF',
  textOnColor: '#FFFFFF',
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
  xxxl: 64,
};

export const Radius = {
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  full: 999,
};

export const FontSize = {
  xs: 11,
  sm: 13,
  md: 16,
  lg: 18,
  xl: 22,
  xxl: 28,
  xxxl: 36,
  hero: 52,
};

export const FontWeight = {
  regular: '400' as const,
  medium: '500' as const,
  semibold: '600' as const,
  bold: '700' as const,
};

export const Shadow = {
  soft: {
    shadowColor: '#1F212D',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  medium: {
    shadowColor: '#1F212D',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 4,
  },
  glow: {
    shadowColor: '#A288FC',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 24,
    elevation: 6,
  },
};
