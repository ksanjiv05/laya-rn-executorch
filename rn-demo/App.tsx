/**
 * Laya × ExecuTorch — on-device demo (screenshot-ready).
 *
 * Monochrome, sharp-corner presentation build. Loads the int8 Laya .pte from the device filesystem
 * and shows each test case as a card: the input state, the typed question, and Laya's on-device
 * decision (choice / score / yes-no) with a calibrated confidence bar. No timing.
 */
import React, {useState} from 'react';
import {
  Platform,
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
// The int8 .pte is pushed to the app's on-device storage, per platform:
//   Android:  adb push laya_xnnpack_int8wo.pte /sdcard/Android/data/<pkg>/files/laya_int8.pte
//   iOS:      copy laya_int8.pte into the app's Documents dir (e.g. via Xcode "Add Files" as a
//             bundle resource + copy on first launch, or `xcrun simctl` push to the container).
const MODEL_PATH = Platform.select({
  android: 'file:///sdcard/Android/data/com.layaexecutorchdemo/files/laya_int8.pte',
  ios: 'laya_int8.pte', // resolved from the app bundle / Documents by the resource fetcher
  default: 'laya_int8.pte',
}) as string;

// Monochrome palette — pure black/white shades.
const C = {
  bg: '#000000',
  card: '#0B0B0B',
  cardHi: '#151515',
  line: '#242424',
  lineHi: '#3A3A3A',
  ink: '#FFFFFF',
  sub: '#9A9A9A',
  faint: '#5E5E5E',
  track: '#161616',
};

const TYPE_TAG: Record<string, string> = {
  choice: 'CHOICE',
  score: 'SCORE',
  noul: 'YES / NO',
};

type Decision = {
  kind: 'choice' | 'score' | 'noul';
  label: string;
  bars: {name: string; p: number}[];
  scoreText?: string;
};

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
const i64 = (arr: number[]) => BigInt64Array.from(arr.map(x => BigInt(x)));

function App(): React.JSX.Element {
  const model = useExecutorchModule({modelSource: MODEL_PATH});
  const [out, setOut] = useState<Record<number, Decision>>({});
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<number | null>(null);

  const decide = async (idx: number): Promise<Decision | null> => {
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
    const res = await model.forward(inputs as any);
    const logits = Array.from(new Float32Array(res[0].dataPtr as ArrayBuffer));
    const k: number = c.k;
    const p = softmax(logits.slice(0, k).map(v => v / c.temperature));

    if (c.question.type === 'choice') {
      const keys: string[] = c.question.criteria;
      const mi = p.indexOf(Math.max(...p));
      return {kind: 'choice', label: keys[mi], bars: keys.map((name, i) => ({name, p: p[i]}))};
    } else if (c.question.type === 'score') {
      const labels: string[] = c.question.criteria;
      const score = p.reduce((a, v, i) => a + v * i, 0);
      const mi = p.indexOf(Math.max(...p));
      return {
        kind: 'score',
        label: labels[mi],
        scoreText: `${score.toFixed(1)} / ${k - 1}`,
        bars: labels.map((name, i) => ({name, p: p[i]})),
      };
    } else {
      const yes = p[1];
      return {
        kind: 'noul',
        label: yes >= 0.5 ? 'Yes' : 'No',
        bars: [
          {name: 'No', p: p[0]},
          {name: 'Yes', p: p[1]},
        ],
      };
    }
  };

  const runOne = async (idx: number) => {
    setBusy(true);
    setActive(idx);
    try {
      const d = await decide(idx);
      if (d) setOut(prev => ({...prev, [idx]: d}));
    } catch {}
    setActive(null);
    setBusy(false);
  };

  const runAll = async () => {
    setBusy(true);
    for (let i = 0; i < DATA.cases.length; i++) {
      setActive(i);
      try {
        const d = await decide(i);
        if (d) setOut(prev => ({...prev, [i]: d}));
      } catch {}
    }
    setActive(null);
    setBusy(false);
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>LAYA</Text>
          <View style={styles.badge}>
            <View style={styles.dot} />
            <Text style={styles.badgeT}>{model.isReady ? 'ON DEVICE' : 'LOADING'}</Text>
          </View>
        </View>
        <View style={styles.rule} />
        <Text style={styles.subtitle}>on-device decision model</Text>

        <Text style={styles.blurb}>
          A small model that <Text style={styles.blurbHi}>decides</Text> instead of chatting — running
          fully offline on this phone. No cloud. No API.
        </Text>

        {/* Cases */}
        {DATA.cases.map((c: any, i: number) => {
          const d = out[i];
          const isActive = active === i;
          return (
            <View key={i} style={styles.card}>
              <View style={styles.cardTop}>
                <View style={styles.typeTag}>
                  <Text style={styles.typeTagT}>{TYPE_TAG[c.question.type]}</Text>
                </View>
                <Text style={styles.caseName}>{c.name.split(' (')[0].toUpperCase()}</Text>
              </View>

              <Text style={styles.fieldLabel}>INPUT</Text>
              <Text style={styles.state}>{c.state}</Text>

              <Text style={styles.fieldLabel}>ASK</Text>
              <Text style={styles.question}>{c.question.instructions}</Text>

              {d ? (
                <View style={styles.result}>
                  <View style={styles.resultHead}>
                    <Text style={styles.decideLabel}>DECISION</Text>
                    <Text style={styles.answer}>
                      {d.label}
                      {d.scoreText ? <Text style={styles.scoreSub}>  ·  {d.scoreText}</Text> : null}
                    </Text>
                  </View>
                  {d.bars.map((b, bi) => {
                    const top = b.p === Math.max(...d.bars.map(x => x.p));
                    return (
                      <View key={bi} style={styles.barRow}>
                        <Text style={[styles.barName, top && styles.barNameTop]}>{b.name}</Text>
                        <View style={styles.barTrack}>
                          <View
                            style={[
                              styles.barFill,
                              {
                                width: `${Math.max(2, b.p * 100)}%`,
                                backgroundColor: top ? C.ink : C.lineHi,
                              },
                            ]}
                          />
                        </View>
                        <Text style={[styles.barPct, top && styles.barNameTop]}>
                          {Math.round(b.p * 100)}%
                        </Text>
                      </View>
                    );
                  })}
                </View>
              ) : (
                <TouchableOpacity
                  style={[styles.runBtn, (!model.isReady || busy) && styles.btnOff]}
                  disabled={!model.isReady || busy}
                  onPress={() => runOne(i)}>
                  <Text style={styles.runBtnT}>{isActive ? 'THINKING…' : 'RUN ON DEVICE'}</Text>
                </TouchableOpacity>
              )}
            </View>
          );
        })}

        <TouchableOpacity
          style={[styles.cta, (!model.isReady || busy) && styles.btnOff]}
          disabled={!model.isReady || busy}
          onPress={runAll}>
          <Text style={styles.ctaT}>{busy ? 'RUNNING…' : 'RUN ALL DECISIONS'}</Text>
        </TouchableOpacity>

        <Text style={styles.footer}>
          MODERNBERT-LARGE · INT8 · EXECUTORCH · REACT NATIVE
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: C.bg},
  content: {padding: 20, paddingBottom: 44, gap: 14},
  header: {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 10},
  title: {color: C.ink, fontSize: 34, fontWeight: '900', letterSpacing: 6},
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    borderColor: C.line,
    borderWidth: 1,
    paddingVertical: 6,
    paddingHorizontal: 11,
  },
  dot: {width: 6, height: 6, backgroundColor: C.ink},
  badgeT: {color: C.sub, fontSize: 10, fontWeight: '800', letterSpacing: 1},
  rule: {height: 1, backgroundColor: C.line, marginTop: 2},
  subtitle: {color: C.faint, fontSize: 12, letterSpacing: 3, textTransform: 'uppercase', marginTop: -6},
  blurb: {color: C.sub, fontSize: 15, lineHeight: 23, marginTop: 2},
  blurbHi: {color: C.ink, fontWeight: '800'},

  card: {backgroundColor: C.card, borderColor: C.line, borderWidth: 1, padding: 18, gap: 7},
  cardTop: {flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 4},
  typeTag: {borderColor: C.lineHi, borderWidth: 1, paddingVertical: 3, paddingHorizontal: 8},
  typeTagT: {color: C.ink, fontSize: 10, fontWeight: '800', letterSpacing: 1.2},
  caseName: {color: C.sub, fontSize: 12, fontWeight: '700', letterSpacing: 2, flexShrink: 1},

  fieldLabel: {color: C.faint, fontSize: 10, fontWeight: '800', letterSpacing: 1.5, marginTop: 6},
  state: {color: C.ink, fontSize: 16, lineHeight: 23},
  question: {color: C.sub, fontSize: 14, lineHeight: 20},

  result: {backgroundColor: C.bg, borderColor: C.line, borderWidth: 1, padding: 14, marginTop: 12, gap: 11},
  resultHead: {flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between'},
  decideLabel: {color: C.faint, fontSize: 10, fontWeight: '800', letterSpacing: 1.5},
  answer: {color: C.ink, fontSize: 24, fontWeight: '900', textTransform: 'capitalize'},
  scoreSub: {fontSize: 14, fontWeight: '700', color: C.sub},
  barRow: {flexDirection: 'row', alignItems: 'center', gap: 12},
  barName: {color: C.faint, fontSize: 12, fontWeight: '600', width: 74, textTransform: 'capitalize'},
  barNameTop: {color: C.ink},
  barTrack: {flex: 1, height: 8, backgroundColor: C.track, overflow: 'hidden'},
  barFill: {height: 8},
  barPct: {color: C.sub, fontSize: 12, fontWeight: '700', width: 38, textAlign: 'right'},

  runBtn: {borderColor: C.lineHi, borderWidth: 1, paddingVertical: 13, alignItems: 'center', marginTop: 10},
  runBtnT: {color: C.ink, fontSize: 13, fontWeight: '800', letterSpacing: 1.5},
  cta: {backgroundColor: C.ink, paddingVertical: 17, alignItems: 'center', marginTop: 4},
  ctaT: {color: C.bg, fontSize: 15, fontWeight: '900', letterSpacing: 1.5},
  btnOff: {opacity: 0.35},
  footer: {color: C.faint, fontSize: 10, textAlign: 'center', letterSpacing: 1.5, marginTop: 8},
});

export default App;
