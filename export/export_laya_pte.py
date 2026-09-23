#!/usr/bin/env python3
"""
Export Laya (ModernBERT-large encoder + custom decision head) to an ExecuTorch .pte
for react-native-executorch, using the XNNPACK CPU backend.

Laya is NOT a stock HF task model, so optimum-executorch's one-command export does not
apply — we rebuild DecisionModel from rl_common.py, load the typed-decisions weights,
wrap forward with STATIC padded shapes (mobile-friendly), then torch.export -> to_edge -> .pte.

Model contract (from rl_agent_api.py / rl_common.py):
  inputs : input_ids[B,L] long, attention_mask[B,L] long, marker_pos[B,K] long,
           marker_mask[B,K] bool, qtype[B] long
  outputs: logits[B,K] float (per-option scores; softmax->probabilities on the JS side),
           act_logits[B,2] float (escalate/answer head)
We fix B=1, L=SEQ_LEN, K=MAX_OPTIONS so the runtime graph is static. The JS side pads/truncates
to these and applies the temperature + softmax post-processing that rl_agent_api does in Python.
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # rl_common.py sits beside this script

def build(model_dir: str):
    from safetensors.torch import load_file
    from rl_common import build_model
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model.encoder.config.reference_compile = False
    model.eval()
    return model, cfg


class ExportWrapper(torch.nn.Module):
    """Static-shape, export-clean forward. Returns (logits, act_logits) as float32."""
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.core(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits.float(), act.float()


def make_example(seq_len: int, max_opts: int, pad_id: int):
    ids = torch.full((1, seq_len), pad_id, dtype=torch.long)
    ids[0, :8] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    att = torch.zeros((1, seq_len), dtype=torch.long); att[0, :8] = 1
    mpos = torch.zeros((1, max_opts), dtype=torch.long); mpos[0, :3] = torch.tensor([2, 4, 6])
    mmask = torch.zeros((1, max_opts), dtype=torch.bool); mmask[0, :3] = True
    qtype = torch.zeros((1,), dtype=torch.long)  # 0=choice
    return (ids, att, mpos, mmask, qtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "model-src", "hf", "typed-decisions"))
    ap.add_argument("--out", default=os.path.join(HERE, "laya_xnnpack.pte"))
    ap.add_argument("--seq-len", type=int, default=192, help="fixed padded sequence length (<= head_max_len keeps it small)")
    ap.add_argument("--max-opts", type=int, default=12, help="fixed max #options (markers) per question")
    ap.add_argument("--sanity", action="store_true", help="run eager vs exported numeric check")
    ap.add_argument("--quantize", action="store_true",
                    help="int8 dynamic PT2E quantization of Linear layers (≈4x smaller, faster CPU)")
    ap.add_argument("--weight-only", action="store_true",
                    help="int8 WEIGHT-ONLY quantization via torchao.quantize_ (≈4x smaller; "
                         "leaves integer gather/index ops untouched — safest for the custom head)")
    ap.add_argument("--backend", choices=["xnnpack", "vulkan", "coreml"], default="xnnpack",
                    help="delegate backend: xnnpack (portable CPU, iOS+Android), "
                         "vulkan (Android GPU), coreml (iOS Neural Engine/GPU — export on macOS only)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    print(f"[1/5] building model from {model_dir}")
    core, cfg = build(model_dir)
    pad_id = 50283  # ModernBERT pad_token_id (from encoder/config.json)

    wrapper = ExportWrapper(core).eval()
    example = make_example(args.seq_len, args.max_opts, pad_id)

    if args.weight_only:
        print("[q] applying int8 weight-only quantization (torchao.quantize_ on Linear)")
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(wrapper, Int8WeightOnlyConfig())

    print(f"[2/5] eager forward (seq_len={args.seq_len}, max_opts={args.max_opts})")
    with torch.no_grad():
        eager_logits, eager_act = wrapper(*example)
    print("      logits", tuple(eager_logits.shape), "act", tuple(eager_act.shape))

    if args.quantize:
        print("[3/5] torch.export + PT2E int8 (dynamic) quantization")
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        training_ep = torch.export.export(wrapper, example, strict=False).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
        prepared = prepare_pt2e(training_ep, quantizer)
        with torch.no_grad():
            prepared(*example)  # observe ranges (dynamic: activation ranges computed at runtime)
        converted = convert_pt2e(prepared)
        with torch.no_grad():
            ep = torch.export.export(converted, example, strict=False)
    else:
        print("[3/5] torch.export (fp32)")
        with torch.no_grad():
            ep = torch.export.export(wrapper, example, strict=False)

    if args.sanity:
        with torch.no_grad():
            gm = ep.module()
            gl, ga = gm(*example)
        err = (gl - eager_logits).abs().max().item()
        print(f"      exported-vs-eager max|Δlogits| = {err:.3e}")

    print(f"[4/5] lower to ExecuTorch + {args.backend}")
    from executorch.exir import to_edge_transform_and_lower, ExecutorchBackendConfig
    if args.backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioner = [XnnpackPartitioner()]
    elif args.backend == "vulkan":
        # Android GPU. Vulkan prefers fp16; note it currently delegates a subset of ops and
        # falls back to portable CPU for the rest, so pair with a real device benchmark.
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        partitioner = [VulkanPartitioner()]
    else:  # coreml — iOS Neural Engine / GPU; real compile needs macOS coremltools runtime
        from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
        partitioner = [CoreMLPartitioner()]
    lowered = to_edge_transform_and_lower(ep, partitioner=partitioner)
    prog = lowered.to_executorch(ExecutorchBackendConfig())

    print(f"[5/5] writing {args.out}")
    with open(args.out, "wb") as f:
        f.write(prog.buffer)
    mb = os.path.getsize(args.out) / 1e6
    print(f"DONE  {args.out}  ({mb:.1f} MB)")
    # emit a runtime contract file the RN package will read
    meta = {
        "seq_len": args.seq_len, "max_opts": args.max_opts, "pad_id": pad_id,
        "inputs": ["input_ids[1,{}]i64".format(args.seq_len), "attention_mask[1,{}]i64".format(args.seq_len),
                   "marker_pos[1,{}]i64".format(args.max_opts), "marker_mask[1,{}]bool".format(args.max_opts),
                   "qtype[1]i64"],
        "outputs": ["logits[1,{}]f32".format(args.max_opts), "act_logits[1,2]f32"],
        "temperature": cfg.get("temperature"), "temperature_by_options": cfg.get("temperature_by_options"),
        "head_max_len": cfg.get("head_max_len"), "max_len": cfg.get("max_len"),
        "qtypes": {"choice": 0, "score": 1, "noul": 2},
    }
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("      wrote", os.path.splitext(args.out)[0] + ".meta.json")


if __name__ == "__main__":
    main()
