/**
 * Laya × ExecuTorch — REAL int8 model, per-case runner.
 *
 * Loads a Laya .pte from the device filesystem and runs each pre-tokenized case in
 * assets/laya_testcases.json individually (tap a case to run just it) or all in sequence.
 * Applies Laya's temperature+softmax, times each forward, compares to the Python reference.
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

const DATA = require('./assets/laya_testcases.json');
const DIR = 'file:///sdcard/Android/data/com.layaexecutorchdemo/files/';
const MODELS = [
  {label: 'int8 · GPU (Vulkan)', path: DIR + 'laya_vulkan_int8.pte'},
  {label: 'int8 · CPU (XNNPACK)', path: DIR + 'laya_int8.pte'},
];

type CaseResult = {summary: string; ms: number; match: boolean};

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
const i64 = (arr: number[]) => BigInt64Array.from(arr.map(x => BigInt(x)));

function App(): React.JSX.Element {
  const [sel, setSel] = useState(0);
  const model = useExecutorchModule({modelSource: MODELS[sel].path});
  const [results, setResults] = useState<Record<number, CaseResult>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const runOne = async (idx: number) => {
    setErr('');
    setBusy(true);
    try {
      const c = DATA.cases[idx];
      const inputs = [
        {dataPtr: i64(c.input_ids), sizes: [1, DATA.seq_len], scalarType: ScalarType.LONG},
        {dataPtr: i64(c.attention_mask), sizes: [1, DATA.seq_len], scalarType: ScalarType.LONG},
        {dataPtr: i64(c.marker_pos), sizes: [1, DATA.max_opts], scalarType: ScalarType.LONG},
        {
          dataPtr: Uint8Array.from(c.marker_mask.map((b: boolean) => (b ? 1 : 0))),
          sizes: [1, DATA.max_opts],
          scalarType: ScalarType.BOOL,
        },
        {dataPtr: i64([c.qtype]), sizes: [1], scalarType: ScalarType.LONG},
      ];
      const t0 = Date.now();
      const res = await model.forward(inputs as any);
      const ms = Date.now() - t0;
      const logits = Array.from(new Float32Array(res[0].dataPtr as ArrayBuffer));
      const k: number = c.k;
      const p = softmax(logits.slice(0, k).map(v => v / c.temperature));

      let summary = '';
      let match = false;
      if (c.question.type === 'choice') {
        const keys: string[] = c.question.criteria;
        const mi = p.indexOf(Math.max(...p));
        summary = `${keys[mi]}  [${p.map(v => v.toFixed(3)).join(', ')}]`;
        match =
          keys[mi] === c.reference.choice &&
          keys.every((kk, i) => Math.abs(p[i] - c.reference.probabilities[kk]) < 0.02);
      } else if (c.question.type === 'score') {
        const score = p.reduce((a, v, i) => a + v * i, 0);
        summary = `score ${score.toFixed(2)} / ${k - 1}  [${p.map(v => v.toFixed(2)).join(', ')}]`;
        match = Math.abs(score - c.reference.score) < 0.05;
      } else {
        summary = `P(true) = ${p[1].toFixed(4)}`;
        match = Math.abs(p[1] - c.reference.noul) < 0.02;
      }
      setResults(prev => ({...prev, [idx]: {summary, ms, match}}));
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
    setBusy(false);
  };

  const runAll = async () => {
    setResults({});
    for (let i = 0; i < DATA.cases.length; i++) {
      await runOne(i);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Laya × ExecuTorch</Text>
        <Text style={styles.subtitle}>per-case runner</Text>

        <View style={styles.sel}>
          {MODELS.map((m, i) => (
            <TouchableOpacity
              key={i}
              disabled={busy}
              onPress={() => {
                setSel(i);
                setResults({});
              }}
              style={[styles.chip, sel === i && styles.chipOn]}>
              <Text style={[styles.chipT, sel === i && styles.chipTOn]}>{m.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.card}>
          <Row label="Selected" value={MODELS[sel].label} />
          <Row label="State" value={model.isReady ? 'ready ✓' : `loading… ${Math.round((model.downloadProgress ?? 0) * 100)}%`} />
        </View>

        {DATA.cases.map((c: any, i: number) => {
          const r = results[i];
          return (
            <View key={i} style={styles.case}>
              <View style={styles.caseHead}>
                <Text style={styles.caseName}>
                  {i + 1}. {c.name}
                </Text>
                {r ? <Text style={r.match ? styles.ok : styles.bad}>{r.match ? '✓' : '✗'}</Text> : null}
              </View>
              {r ? (
                <>
                  <Text style={styles.caseOut}>{r.summary}</Text>
                  <Text style={styles.caseMs}>{r.ms} ms</Text>
                </>
              ) : null}
              <TouchableOpacity
                style={[styles.runBtn, (!model.isReady || busy) && styles.btnOff]}
                disabled={!model.isReady || busy}
                onPress={() => runOne(i)}>
                <Text style={styles.runBtnT}>Run case {i + 1}</Text>
              </TouchableOpacity>
            </View>
          );
        })}

        <TouchableOpacity
          style={[styles.btn, (!model.isReady || busy) && styles.btnOff]}
          disabled={!model.isReady || busy}
          onPress={runAll}>
          <Text style={styles.btnText}>{busy ? 'Running…' : 'Run all in sequence'}</Text>
        </TouchableOpacity>

        {err ? <Text style={styles.bad}>{err}</Text> : null}
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
  content: {padding: 22, gap: 14},
  title: {color: '#e2e8f0', fontSize: 25, fontWeight: '800', marginTop: 10},
  subtitle: {color: '#64748b', fontSize: 13, marginBottom: 4},
  sel: {flexDirection: 'row', flexWrap: 'wrap', gap: 8},
  chip: {borderWidth: 1, borderColor: '#334155', borderRadius: 20, paddingVertical: 8, paddingHorizontal: 14},
  chipOn: {backgroundColor: '#38bdf8', borderColor: '#38bdf8'},
  chipT: {color: '#94a3b8', fontSize: 12, fontWeight: '600'},
  chipTOn: {color: '#04121f'},
  card: {backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, borderRadius: 14, padding: 15, gap: 6},
  row: {flexDirection: 'row', justifyContent: 'space-between'},
  rowL: {color: '#64748b', fontSize: 13},
  rowV: {color: '#e2e8f0', fontSize: 13, fontWeight: '600'},
  case: {backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, borderRadius: 12, padding: 13, gap: 6},
  caseHead: {flexDirection: 'row', justifyContent: 'space-between'},
  caseName: {color: '#94a3b8', fontSize: 13, fontWeight: '600', flexShrink: 1},
  caseOut: {color: '#38bdf8', fontSize: 14, fontFamily: 'monospace'},
  caseMs: {color: '#f59e0b', fontSize: 12, fontFamily: 'monospace'},
  runBtn: {backgroundColor: '#1e293b', borderRadius: 10, paddingVertical: 10, alignItems: 'center', marginTop: 2},
  runBtnT: {color: '#e2e8f0', fontSize: 13, fontWeight: '700'},
  btn: {backgroundColor: '#38bdf8', borderRadius: 12, paddingVertical: 15, alignItems: 'center'},
  btnOff: {opacity: 0.4},
  btnText: {color: '#04121f', fontSize: 16, fontWeight: '800'},
  ok: {color: '#34d399', fontSize: 16, fontWeight: '800'},
  bad: {color: '#f87171', fontSize: 14, fontWeight: '700'},
});

export default App;
