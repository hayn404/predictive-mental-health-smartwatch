// Learn more: https://docs.expo.dev/guides/customizing-metro/
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Bundle on-device model files (.tflite for sleep via react-native-fast-tflite;
// .onnx kept for reference) as assets so `require(...)` resolves them.
// .wasm is needed for expo-sqlite's web build (wa-sqlite) to load its binary.
for (const ext of ['tflite', 'onnx', 'wasm']) {
  if (!config.resolver.assetExts.includes(ext)) config.resolver.assetExts.push(ext);
}

// expo-sqlite's web worker relies on SharedArrayBuffer, which browsers only
// allow on cross-origin-isolated pages -- these headers turn that on for the
// dev server.
config.server.enhanceMiddleware = (middleware) => {
  return (req, res, next) => {
    res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    return middleware(req, res, next);
  };
};

module.exports = config;
