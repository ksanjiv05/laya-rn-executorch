const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {};

const merged = mergeConfig(getDefaultConfig(__dirname), config);
// Bundle ExecuTorch model binaries as assets so `require('./assets/models/x.pte')` resolves.
merged.resolver.assetExts.push('pte', 'ptl');

module.exports = merged;
