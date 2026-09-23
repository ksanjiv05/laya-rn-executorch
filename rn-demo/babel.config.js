module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // react-native-executorch pulls in react-native-worklets; its babel plugin must be LAST.
  plugins: ['react-native-worklets/plugin'],
};
