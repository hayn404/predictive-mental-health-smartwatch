// Learn more: https://docs.expo.dev/guides/customizing-metro/
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Bundle on-device model files (.tflite for sleep via react-native-fast-tflite;
// .onnx kept for reference) as assets so `require(...)` resolves them.
for (const ext of ['tflite', 'onnx']) {
  if (!config.resolver.assetExts.includes(ext)) config.resolver.assetExts.push(ext);
}

module.exports = config;
