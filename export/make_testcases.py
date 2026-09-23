#!/usr/bin/env python3
"""
Generate MULTIPLE fixed test cases for the on-device int8 Laya .pte and dump them (with Python
int8 reference answers + tokenized static-shape tensors) to rn-demo/assets/laya_testcases.json.

Covers all three question types (choice / score / noul) and varied state lengths so we can analyze
both correctness (vs reference) and inference time on device.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_common import build_model, build_sequence, render_options, QTYPES, temp_bucket

MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "model-src", "hf", "typed-decisions"))
SEQ_LEN, MAX_OPTS, PAD_ID = 192, 12, 50283

CASES = [
    {
        "name": "sentiment (choice, 3-way)",
        "state": "The delivery was two days late and the box arrived crushed, but the item itself works perfectly.",
        "question": {"type": "choice",
                     "instructions": "Classify the customer's overall sentiment about this order.",
                     "criteria": ["positive", "neutral", "negative"]},
    },
    {
        "name": "intent routing (choice, 4-way)",
        "state": "Hi, I was charged twice for my subscription this month and need one of the payments reversed.",
        "question": {"type": "choice",
                     "instructions": "Which team should handle this support ticket?",
                     "criteria": ["billing", "technical", "sales", "general"]},
    },
    {
        "name": "urgency (score, 1-5)",
        "state": "Our production database is down, customers cannot log in, and we are losing revenue every minute.",
        "question": {"type": "score",
                     "instructions": "Rate the urgency of this message from lowest to highest.",
                     "criteria": ["not urgent", "low", "medium", "high", "critical"]},
    },
    {
        "name": "policy check (noul, yes/no)",
        "state": "The email asks the user to click a link and enter their bank password to 'verify' their account.",
        "question": {"type": "noul",
                     "instructions": "Is this message a phishing attempt?",
                     "criteria": None},
    },
]


def to_internal(q):
    t = q["type"]; crit = q.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": t, "ins": q["instructions"], "crit": crit}


def build_case(tok, cfg, case):
    q = to_internal(case["question"])
    ids, markers = build_sequence(tok, case["state"], q, cfg["max_len"], cfg["head_max_len"])
    k = len(markers)
    assert k <= MAX_OPTS and len(ids) <= SEQ_LEN, (case["name"], len(ids), k)
    return {
        "input_ids": ids + [PAD_ID] * (SEQ_LEN - len(ids)),
        "attention_mask": [1] * len(ids) + [0] * (SEQ_LEN - len(ids)),
        "marker_pos": markers + [0] * (MAX_OPTS - k),
        "marker_mask": [True] * k + [False] * (MAX_OPTS - k),
        "qtype": QTYPES[q["t"]], "k": k, "ntok": len(ids),
    }


def main():
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    with open(os.path.join(MODEL_DIR, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(MODEL_DIR, "encoder"))
    model.load_state_dict(load_file(os.path.join(MODEL_DIR, "model.safetensors")), strict=True)
    model.encoder.config.reference_compile = False
    model.eval()
    quantize_(model, Int8WeightOnlyConfig())

    out_cases = []
    for case in CASES:
        b = build_case(tok, cfg, case)
        t = {n: torch.tensor([b[n]], dtype=(torch.bool if n == "marker_mask" else torch.long))
             for n in ("input_ids", "attention_mask", "marker_pos", "marker_mask")}
        qtype = b["qtype"]
        t0 = time.time()
        with torch.no_grad():
            logits, act = model(t["input_ids"], t["attention_mask"], t["marker_pos"],
                                t["marker_mask"], torch.tensor([qtype], dtype=torch.long))
        cpu_ms = round((time.time() - t0) * 1000)
        logits = logits.float().numpy()[0]
        k = b["k"]
        temp = cfg.get("temperature_by_options", {}).get(temp_bucket(qtype, k),
                                                         cfg.get("temperature", [1, 1, 1])[qtype])
        z = logits[:k] / temp
        p = np.exp(z - z.max()); p = p / p.sum()

        crit = case["question"].get("criteria")
        if case["question"]["type"] == "choice":
            ref = {"choice": crit[int(p.argmax())],
                   "probabilities": {c: round(float(v), 4) for c, v in zip(crit, p)}}
        elif case["question"]["type"] == "score":
            ref = {"score": round(float((np.arange(k) * p).sum()), 4),
                   "legend": {str(i): c for i, c in enumerate(crit)},
                   "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}}
        else:
            ref = {"noul": round(float(p[1]), 4)}

        out_cases.append({
            "name": case["name"], "state": case["state"], "question": case["question"],
            "ntok": b["ntok"], "k": k, "qtype": qtype, "temperature": float(temp),
            "input_ids": b["input_ids"], "attention_mask": b["attention_mask"],
            "marker_pos": b["marker_pos"], "marker_mask": [bool(x) for x in b["marker_mask"]],
            "reference": ref, "cpu_desktop_ms": cpu_ms,
        })
        print(f"[{case['name']}] ntok={b['ntok']} k={k} desktop={cpu_ms}ms  ref={json.dumps(ref)}")

    dst = os.path.abspath(os.path.join(HERE, "..", "rn-demo", "assets", "laya_testcases.json"))
    with open(dst, "w") as f:
        json.dump({"seq_len": SEQ_LEN, "max_opts": MAX_OPTS, "cases": out_cases}, f, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    main()
