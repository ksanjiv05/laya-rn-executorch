# laya-for-react-native

Bring **Laya** — the open-source, Jev-compatible **System-1 decision model** — to React Native.

> ⚠️ Status: project scaffolding. This README first explains *what Laya is* and *how it compares to Jev*
> (the research the build is based on), then the plan for the RN package.

---

## 1. What is Laya?

**Laya is a *decision model*, not a chat model.** It does **not** generate text. You hand it:
- a **state** — some JSON your app already has (a support ticket, an email, a log line, a tool call, a retrieved passage), and
- one or more **typed questions** whose answers are bounded in advance,

and in **one forward pass** it returns a **machine-usable answer with a calibrated probability** for each question. No paragraph to parse, no token-by-token generation.

It answers three kinds of typed question:
| Type | What you get back |
|---|---|
| **choice** | pick exactly one of N options, with a probability per option |
| **score** | an expected level on an ordered rubric (e.g. 1–5), with the full distribution |
| **noul** (bool) | a calibrated `P(true)` for a yes/no statement |

This is the "**System 1**" idea from Daniel Kahneman: fast, intuitive, automatic decisions — *pick immediately from defined options* — as opposed to an LLM's slow, deliberate, "System 2" token-by-token reasoning.

**Who makes it:** Convai Innovations. Released **September 2026**, **Apache-2.0**, three open checkpoints:
- **421M** English base — built on **ModernBERT-large**
- **322M** multilingual — built on **mmBERT-base**, 100+ languages
- a **typed-decisions** fine-tuned checkpoint

**How it runs (relevant to us):** the community package [`@receptron/laya`](https://github.com/receptron/laya) runs the model from **Node.js / TypeScript via ONNX Runtime** — **no PyTorch, no Python at runtime**. There's an ONNX export at `receptron/laya-onnx` on Hugging Face (ModernBERT-large encoder + Laya's decision head).

### What it's good at
Classification, routing, scoring, ranking, safety/quality gates, extraction-of-a-typed-value — anything where the set of possible answers is known ahead of time and you want a **confidence number** attached.

### What it is NOT
Not a generator. It can't write a reply, summarize freely, or hold a conversation. For those you still reach for an LLM (System 2). The intended pattern is **hybrid**: Laya makes the fast typed decisions / routing, an LLM handles the open-ended generation only when needed.

---

## 2. Laya vs Jev

**Jev** (by **TypeSafe AI**, announced **15 September 2026**) is the model that *defined* this "System One / typed-decision" category — same interface: send state + typed questions, get back typed answers with probabilities. Jev is **hosted/closed**: strong published latency & calibration, priced per token, but **no open weights, no parameter count, no self-hosting**.

**Laya is the open-source answer to Jev** — "Jev-compatible," so it targets the same request/response shape, but ships weights you can run yourself.

### Why Laya can be "better" than Jev
It depends on what you value — the honest picture from the benchmark coverage:

| Dimension | Laya (Convai) | Jev (TypeSafe) |
|---|---|---|
| **License / weights** | ✅ **Apache-2.0, open weights**, self-hostable | ❌ Closed, hosted-only |
| **Cost** | ✅ Run it yourself — no per-token fee | Per-token ($ ~0.042 / M input tokens) |
| **Latency (p50)** | ✅ **~32.8 ms** on a Tesla T4 (multilingual) | ~236–276 ms |
| **Speedup** | **~7.8×** faster (attributed to non-autoregressive scoring) | baseline |
| **Accuracy (typed-decisions)** | **0.766** on the card's checkpoint | 0.727 (Jev 1.13.0) |
| **Calibration (ECE)** | ~0.081 expected calibration error after temperature fitting | strong, published |
| **Multilingual** | ✅ 322M variant, 100+ languages | — |
| **Runtime** | ONNX Runtime (Node/TS/native), local-first | API only |

**So "better than Jev" mainly means:** open & self-hostable (privacy + zero inference cost), dramatically lower latency, and a higher *reported* accuracy on its specialised checkpoint — plus first-class multilingual.

### The honest caveats (from the skeptical write-ups)
- The headline **0.766 accuracy is from a checkpoint fine-tuned on the benchmark's own training split**; the **zero-shot** number is lower. Jev's 0.727 was measured differently, so it's **not a clean apples-to-apples** comparison.
- Laya is a **credible open candidate, not a proven universal Jev replacement** — validate on *your own* task before trusting the leaderboard.
- The ~33 ms figure is on a **T4 GPU**; on a phone/CPU it will be higher (still far cheaper than a generative LLM call).

**Bottom line:** Laya is the open, local-first, low-latency, calibrated typed-decision engine; Jev is the polished hosted incumbent that created the category. For an on-device / cost-sensitive / privacy-first app, Laya is the natural pick — with the caveat that you should benchmark it on your real data rather than take the marketing accuracy at face value.

---

## 3. Plan for this package (`laya-for-react-native`)

Goal: let a React Native app ask Laya typed questions (`choice` / `score` / `noul`) and get back typed answers with calibrated probabilities — with a clean TS API mirroring `@receptron/laya`.

**Two viable runtimes (to be decided — see below):**
- **A. On-device** via `onnxruntime-react-native` (v1.24.3): the ModernBERT ONNX + a JS/native tokenizer run *on the phone*. Fully offline, private, zero inference cost — matches a privacy-first app. Cost: model size on mobile + tokenizer wiring.
- **B. Thin client + Node service** running `@receptron/laya` server-side: tiny app, uses the official package directly, but needs a backend and isn't offline.

_Toolchain present:_ node v24.21, npm 11.19; `@receptron/laya@0.1.2` and `onnxruntime-react-native@1.24.3` both on npm.

_(Architecture, module layout, and API surface land here once the runtime is chosen.)_

---

### Sources
Convai model card & site, `receptron/laya` (GitHub) + `receptron/laya-onnx` (HF), TypeSafe/Jev docs, and independent benchmark write-ups (Flowtivity, Wavect, AI Weekly, eesel, alphamatch). September 2026.
