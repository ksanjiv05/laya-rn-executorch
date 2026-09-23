/**
 * Laya × ExecuTorch — REAL int8 model, multi-case benchmark.
 *
 * Loads the 603 MB int8 Laya .pte from the device filesystem, runs every pre-tokenized case in
 * assets/laya_testcases.json (choice / score / noul), applies Laya's temperature+softmax
 * post-processing, times each forward pass, and compares to the Python int8 reference.
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
  {label: 'int8 · CPU (XNNPACK)', path: DIR + 'laya_int8.pte'},
  {label: 'int8 · GPU (Vulkan)', path: DIR + 'laya_vulkan_int8.pte'},
];

type CaseResult = {
  name: string;
  summary: string;
  ms: number;
  match: boolean;
};

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function i64(arr: number[]) {
  return BigInt64Array.from(arr.map(x => BigInt(x)));
}

function App(): React.JSX.Element {
  const [sel, setSel] = useState(0);
  const model = useExecutorchModule({modelSource: MODELS[sel].path});
  const [results, setResults] = useState<CaseResult[]>([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState('');

  const runAll = async () => {
    setErr('');
    setRunning(true);
    setResults([]);
    const acc: CaseResult[] = [];
    try {
      for (const c of DATA.cases) {
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
        const z = logits.slice(0, k).map(v => v / c.temperature);
        const p = softmax(z);

        let summary = '';
        let match = false;
        if (c.question.type === 'choice') {
          const keys: string[] = c.question.criteria;
          const idx = p.indexOf(Math.max(...p));
          summary = `${keys[idx]}  [${p.map(v => v.toFixed(3)).join(', ')}]`;
          match =
            keys[idx] === c.reference.choice &&
            keys.every(
              (kk, i) => Math.abs(p[i] - c.reference.probabilities[kk]) < 0.02,
            );
        } else if (c.question.type === 'score') {
          const score = p.reduce((a, v, i) => a + v * i, 0);
          summary = `score ${score.toFixed(2)} / ${k - 1}  [${p
            .map(v => v.toFixed(2))
            .join(', ')}]`;
          match = Math.abs(score - c.reference.score) < 0.05;
        } else {
          const pt = p[1];
          summary = `P(true) = ${pt.toFixed(4)}`;
          match = Math.abs(pt - c.reference.noul) < 0.02;
        }
        acc.push({name: c.name, summary, ms, match});
        setResults([...acc]);
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    }
    setRunning(false);
  };

  const times = results.map(r => r.ms);
  const avg = times.length ? Math.round(times.reduce((a, b) => a + b, 0) / times.length) : 0;
  const passed = results.filter(r => r.match).length;

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Laya × ExecuTorch</Text>
        <Text style={styles.subtitle}>backend benchmark · {DATA.cases.length} cases</Text>

        <View style={styles.sel}>
          {MODELS.map((m, i) => (
            <TouchableOpacity
              key={i}
              disabled={running}
              onPress={() => {
                setSel(i);
                setResults([]);
              }}
              style={[styles.chip, sel === i && styles.chipOn]}>
              <Text style={[styles.chipT, sel === i && styles.chipTOn]}>{m.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.card}>
          <Row label="Selected" value={MODELS[sel].label} />
          <Row label="Encoder" value="ModernBERT-large + head" />
          <Row
            label="State"
            value={
              model.isReady
                ? 'ready ✓'
                : `loading… ${Math.round((model.downloadProgress ?? 0) * 100)}%`
            }
          />
        </View>

        <TouchableOpacity
          style={[styles.btn, (!model.isReady || running) && styles.btnOff]}
          disabled={!model.isReady || running}
          onPress={runAll}>
          <Text style={styles.btnText}>
            {running ? 'Running…' : `Run all ${DATA.cases.length} cases`}
          </Text>
        </TouchableOpacity>

        {results.map((r, i) => (
          <View key={i} style={styles.case}>
            <View style={styles.caseHead}>
              <Text style={styles.caseName}>{r.name}</Text>
              <Text style={r.match ? styles.ok : styles.bad}>{r.match ? '✓' : '✗'}</Text>
            </View>
            <Text style={styles.caseOut}>{r.summary}</Text>
            <Text style={styles.caseMs}>{r.ms} ms</Text>
          </View>
        ))}

        {results.length === DATA.cases.length ? (
          <View style={styles.card}>
            <Text style={styles.k}>SUMMARY</Text>
            <Row label="Matched reference" value={`${passed} / ${results.length}`} />
            <Row label="Avg inference" value={`${avg} ms`} />
            <Row
              label="Min / Max"
              value={`${Math.min(...times)} / ${Math.max(...times)} ms`}
            />
          </View>
        ) : null}

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
  chip: {
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 14,
  },
  chipOn: {backgroundColor: '#38bdf8', borderColor: '#38bdf8'},
  chipT: {color: '#94a3b8', fontSize: 12, fontWeight: '600'},
  chipTOn: {color: '#04121f'},
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
  rowV: {color: '#e2e8f0', fontSize: 13, fontWeight: '600'},
  k: {color: '#64748b', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1},
  btn: {backgroundColor: '#38bdf8', borderRadius: 12, paddingVertical: 15, alignItems: 'center'},
  btnOff: {backgroundColor: '#334155'},
  btnText: {color: '#04121f', fontSize: 16, fontWeight: '800'},
  case: {
    backgroundColor: '#0f172a',
    borderColor: '#1e293b',
    borderWidth: 1,
    borderRadius: 12,
    padding: 13,
    gap: 4,
  },
  caseHead: {flexDirection: 'row', justifyContent: 'space-between'},
  caseName: {color: '#94a3b8', fontSize: 13, fontWeight: '600'},
  caseOut: {color: '#38bdf8', fontSize: 14, fontFamily: 'monospace'},
  caseMs: {color: '#f59e0b', fontSize: 12, fontFamily: 'monospace'},
  ok: {color: '#34d399', fontSize: 16, fontWeight: '800'},
  bad: {color: '#f87171', fontSize: 14, fontWeight: '700'},
});

export default App;
