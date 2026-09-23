/**
 * @format
 */

import { AppRegistry } from 'react-native';
// react-native-executorch 0.10.x requires a ResourceFetcher adapter registered ONCE
// before any model loads. In bare RN that adapter is the bare-resource-fetcher package.
import { initExecutorch } from 'react-native-executorch/legacy';
import { BareResourceFetcher } from 'react-native-executorch-bare-resource-fetcher';
import App from './App';
import { name as appName } from './app.json';

initExecutorch({ resourceFetcher: BareResourceFetcher });

AppRegistry.registerComponent(appName, () => App);
