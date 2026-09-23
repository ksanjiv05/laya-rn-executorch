/**
 * Laya × ExecuTorch — bare-minimum smoke test.
 *
 * Proves the react-native-executorch runtime loads a .pte and runs a forward pass on-device.
 * Uses a tiny model (y = relu(Wx+b), 4->3) so the build stays light; the same ExecutorchModule
 * bindings drive the full Laya model once we ship its .pte.
 */
import React, {useState} from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  useExecutorchModule,
  ScalarType,
} from 'react-native-executorch/legacy';

function App(): React.JSX.Element {
  const model = useExecutorchModule({
    modelSource: require('./assets/models/tiny.pte'),
  });

  const [output, setOutput] = useState<string>('—');
  const [err, setErr] = useState<string>('');

  const run = async () => {
    setErr('');
    try {
      const input = {
        dataPtr: new Float32Array([1.0, -2.0, 3.0, -4.0]),
        sizes: [1, 4],
        scalarType: ScalarType.FLOAT,
      };
      const t0 = Date.now();
      const out = await model.forward([input]);
      const ms = Date.now() - t0;
      const arr = Array.from(new Float32Array(out[0].dataPtr as ArrayBuffer));
      setOutput(`[${arr.map(v => v.toFixed(4)).join(', ')}]  (${ms} ms)`);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Laya × ExecuTorch</Text>
        <Text style={styles.subtitle}>on-device inference smoke test</Text>

        <View style={styles.card}>
          <Row label="Runtime" value="react-native-executorch" />
          <Row label="Model" value="tiny.pte (Linear 4→3 + ReLU)" />
          <Row
            label="State"
            value={
              model.isReady
                ? 'ready ✓'
                : model.isGenerating
                ? 'working…'
                : `loading… ${Math.round((model.downloadProgress ?? 0) * 100)}%`
            }
          />
        </View>

        <TouchableOpacity
          style={[styles.btn, !model.isReady && styles.btnDisabled]}
          disabled={!model.isReady}
          onPress={run}>
          <Text style={styles.btnText}>Run forward pass</Text>
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.outLabel}>input</Text>
          <Text style={styles.outValue}>[1.0, -2.0, 3.0, -4.0]</Text>
          <Text style={styles.outLabel}>output</Text>
          <Text style={styles.outValue}>{output}</Text>
          {err ? <Text style={styles.err}>{err}</Text> : null}
        </View>

        <Text style={styles.note}>
          If the output array renders after tapping, the ExecuTorch runtime loaded the
          .pte and executed it natively on this device.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Row({label, value}: {label: string; value: string}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#0b1220'},
  content: {padding: 24, gap: 18},
  title: {color: '#e2e8f0', fontSize: 26, fontWeight: '800', marginTop: 12},
  subtitle: {color: '#64748b', fontSize: 14, marginBottom: 8},
  card: {
    backgroundColor: '#0f172a',
    borderColor: '#1e293b',
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    gap: 8,
  },
  row: {flexDirection: 'row', justifyContent: 'space-between'},
  rowLabel: {color: '#64748b', fontSize: 13},
  rowValue: {color: '#e2e8f0', fontSize: 13, fontWeight: '600'},
  btn: {
    backgroundColor: '#38bdf8',
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
  },
  btnDisabled: {backgroundColor: '#334155'},
  btnText: {color: '#04121f', fontSize: 16, fontWeight: '800'},
  outLabel: {
    color: '#64748b',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: 4,
  },
  outValue: {color: '#38bdf8', fontSize: 15, fontFamily: 'monospace'},
  err: {color: '#f87171', fontSize: 13, marginTop: 8},
  note: {color: '#475569', fontSize: 12, lineHeight: 18},
});

export default App;
