#!/usr/bin/env python3
"""
Generate a fixed test case for the on-device int8 Laya .pte:
  - tokenize a (state, question) with the REAL tokenizer + build_sequence,
  - pad to the exported static shapes (seq_len, max_opts),
  - run the int8 weight-only model to get the REFERENCE logits + answer,
  - dump everything to rn-demo/assets/laya_testcase.json for the RN app to replay.

The app feeds the SAME padded tensors to the on-device .pte and must reproduce the answer.
"""
import json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_common import build_model, build_sequence, render_options, QTYPES, temp_bucket

MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "model-src", "hf", "typed-decisions"))
SEQ_LEN, MAX_OPTS, PAD_ID = 192, 12, 50283

# ---- the test example (a 3-way sentiment CHOICE) ----
STATE = "The delivery was two days late and the box arrived crushed, but the item itself works perfectly."
QUESTION = {
    "type": "choice",
    "instructions": "Classify the customer's overall sentiment about this order.",
    "criteria": ["positive", "neutral", "negative"],
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
    quantize_(model, Int8WeightOnlyConfig())  # match the on-device int8 build

    q = {"t": QUESTION["type"], "ins": QUESTION["instructions"],
         "crit": {c: None for c in QUESTION["criteria"]}}
    ids, markers = build_sequence(tok, STATE, q, cfg["max_len"], cfg["head_max_len"])
    k = len(markers)
    assert k <= MAX_OPTS and len(ids) <= SEQ_LEN, (len(ids), k)

    # pad to static export shapes
    input_ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))
    attention_mask = [1] * len(ids) + [0] * (SEQ_LEN - len(ids))
    marker_pos = markers + [0] * (MAX_OPTS - k)
    marker_mask = [True] * k + [False] * (MAX_OPTS - k)
    qtype = QTYPES[q["t"]]

    t = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "marker_pos": torch.tensor([marker_pos], dtype=torch.long),
        "marker_mask": torch.tensor([marker_mask], dtype=torch.bool),
        "qtype": torch.tensor([qtype], dtype=torch.long),
    }
    with torch.no_grad():
        logits, act = model(t["input_ids"], t["attention_mask"], t["marker_pos"],
                            t["marker_mask"], t["qtype"])
    logits = logits.float().numpy()[0]

    # post-process exactly like rl_agent_api.system_one (choice)
    import numpy as np
    temp = cfg.get("temperature_by_options", {}).get(temp_bucket(qtype, k),
                                                     cfg.get("temperature", [1, 1, 1])[qtype])
    z = logits[:k] / temp
    p = np.exp(z - z.max()); p = p / p.sum()
    keys = QUESTION["criteria"]
    answer = {
        "choice": keys[int(p.argmax())],
        "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
    }

    out = {
        "state": STATE, "question": QUESTION,
        "seq_len": SEQ_LEN, "max_opts": MAX_OPTS, "k": k, "qtype": qtype,
        "temperature": float(temp),
        "input_ids": input_ids, "attention_mask": attention_mask,
        "marker_pos": marker_pos, "marker_mask": [bool(b) for b in marker_mask],
        "reference_logits_k": [round(float(x), 5) for x in logits[:k]],
        "reference_answer": answer,
    }
    dst = os.path.abspath(os.path.join(HERE, "..", "rn-demo", "assets", "laya_testcase.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dst)
    print("tokens:", len(ids), "markers(k):", k, "temp:", round(temp, 4))
    print("REFERENCE ANSWER:", json.dumps(answer))

if __name__ == "__main__":
    main()
