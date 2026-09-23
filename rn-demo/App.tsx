/**
 * Laya × ExecuTorch — REAL model on-device test.
 *
 * Loads the int8 weight-only Laya .pte (ModernBERT-large encoder + typed-decision head, 603 MB)
 * from the device filesystem, replays a pre-tokenized test case (assets/laya_testcase.json),
 * runs the forward pass natively, applies Laya's temperature+softmax post-processing, and
 * compares the on-device answer to the Python reference baked into the test case.
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
import {useExecutorchModule, ScalarType} from 'react-native-executorch/legacy';

const tc = require('./assets/laya_testcase.json');

// The .pte was pushed here with:
//   adb push laya_xnnpack_int8wo.pte /sdcard/Android/data/com.layaexecutorchdemo/files/laya_int8.pte
const MODEL_PATH =
  'file:///sdcard/Android/data/com.layaexecutorchdemo/files/laya_int8.pte';

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function App(): React.JSX.Element {
  const model = useExecutorchModule({modelSource: MODEL_PATH});
  const [out, setOut] = useState<string>('—');
  const [verdict, setVerdict] = useState<string>('');
  const [err, setErr] = useState<string>('');

  const run = async () => {
    setErr('');
    setVerdict('');
    try {
      const inputs = [
        {
          dataPtr: BigInt64Array.from(tc.input_ids.map((x: number) => BigInt(x))),
          sizes: [1, tc.seq_len],
          scalarType: ScalarType.LONG,
        },
        {
          dataPtr: BigInt64Array.from(tc.attention_mask.map((x: number) => BigInt(x))),
          sizes: [1, tc.seq_len],
          scalarType: ScalarType.LONG,
        },
        {
          dataPtr: BigInt64Array.from(tc.marker_pos.map((x: number) => BigInt(x))),
          sizes: [1, tc.max_opts],
          scalarType: ScalarType.LONG,
        },
        {
          // bool tensor packed as bytes (0/1)
          dataPtr: Uint8Array.from(tc.marker_mask.map((b: boolean) => (b ? 1 : 0))),
          sizes: [1, tc.max_opts],
          scalarType: ScalarType.BOOL,
        },
        {
          dataPtr: BigInt64Array.from([BigInt(tc.qtype)]),
          sizes: [1],
          scalarType: ScalarType.LONG,
        },
      ];

      const t0 = Date.now();
      const res = await model.forward(inputs as any);
      const ms = Date.now() - t0;

      // output[0] = logits [1, max_opts]; take the k active options
      const logitsAll = Array.from(new Float32Array(res[0].dataPtr as ArrayBuffer));
      const k: number = tc.k;
      const z = logitsAll.slice(0, k).map(v => v / tc.temperature);
      const probs = softmax(z);
      const keys: string[] = tc.question.criteria;
      const choiceIdx = probs.indexOf(Math.max(...probs));

      const line = keys
        .map((kk, i) => `${kk}: ${probs[i].toFixed(4)}`)
        .join('   ');
      setOut(`choice = ${keys[choiceIdx]}\n${line}\n(${ms} ms on device)`);

      const refChoice = tc.reference_answer.choice;
      const match =
        keys[choiceIdx] === refChoice &&
        keys.every(
          kk =>
            Math.abs(
              probs[keys.indexOf(kk)] - tc.reference_answer.probabilities[kk],
            ) < 0.02,
        );
      setVerdict(
        match
          ? `✓ MATCHES Python reference (${refChoice})`
          : `✗ differs from reference (${refChoice})`,
      );
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Laya × ExecuTorch</Text>
        <Text style={styles.subtitle}>real int8 model · on-device typed decision</Text>

        <View style={styles.card}>
          <Text style={styles.k}>STATE</Text>
          <Text style={styles.v}>{tc.state}</Text>
          <Text style={styles.k}>QUESTION ({tc.question.type})</Text>
          <Text style={styles.v}>{tc.question.instructions}</Text>
          <Text style={styles.k}>OPTIONS</Text>
          <Text style={styles.v}>{tc.question.criteria.join(' · ')}</Text>
        </View>

        <View style={styles.card}>
          <Row label="Model" value="Laya int8-WO (603 MB)" />
          <Row label="Encoder" value="ModernBERT-large + decision head" />
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
          style={[styles.btn, !model.isReady && styles.btnOff]}
          disabled={!model.isReady}
          onPress={run}>
          <Text style={styles.btnText}>Run typed decision</Text>
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.k}>ON-DEVICE OUTPUT</Text>
          <Text style={styles.out}>{out}</Text>
          {verdict ? (
            <Text style={[styles.verdict, verdict.startsWith('✓') ? styles.ok : styles.bad]}>
              {verdict}
            </Text>
          ) : null}
          {err ? <Text style={styles.bad}>{err}</Text> : null}
        </View>

        <Text style={styles.note}>
          Reference (Python int8): {tc.reference_answer.choice} · pos{' '}
          {tc.reference_answer.probabilities.positive}
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function Row({label, value}: {label: string; value: string}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowL}>{label}</Text>
      <Text style={styles.rowV}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: '#0b1220'},
  content: {padding: 22, gap: 16},
  title: {color: '#e2e8f0', fontSize: 25, fontWeight: '800', marginTop: 10},
  subtitle: {color: '#64748b', fontSize: 13, marginBottom: 6},
  card: {
    backgroundColor: '#0f172a',
    borderColor: '#1e293b',
    borderWidth: 1,
    borderRadius: 14,
    padding: 15,
    gap: 6,
  },
  row: {flexDirection: 'row', justifyContent: 'space-between'},
  rowL: {color: '#64748b', fontSize: 13},
  rowV: {color: '#e2e8f0', fontSize: 13, fontWeight: '600', flexShrink: 1, textAlign: 'right'},
  k: {color: '#64748b', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginTop: 4},
  v: {color: '#cbd5e1', fontSize: 14, lineHeight: 20},
  btn: {backgroundColor: '#38bdf8', borderRadius: 12, paddingVertical: 15, alignItems: 'center'},
  btnOff: {backgroundColor: '#334155'},
  btnText: {color: '#04121f', fontSize: 16, fontWeight: '800'},
  out: {color: '#38bdf8', fontSize: 15, fontFamily: 'monospace', lineHeight: 22},
  verdict: {fontSize: 14, fontWeight: '700', marginTop: 8},
  ok: {color: '#34d399'},
  bad: {color: '#f87171'},
  note: {color: '#475569', fontSize: 12},
});

export default App;
