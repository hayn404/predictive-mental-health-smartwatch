// Learn more: https://docs.expo.dev/guides/customizing-metro/
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Bundle the on-device sleep model as an asset so
// `require('@/assets/ml/sleep/sleep_stage_model.onnx')` resolves.
if (!config.resolver.assetExts.includes('onnx')) {
  config.resolver.assetExts.push('onnx');
}

module.exports = config;
