# Core ML export notes (iOS Neural Engine / GPU)

Laya runs on iOS **today** via the XNNPACK `.pte` (`laya_xnnpack_int8wo.pte`) — that is the recommended
iOS build and needs no Mac-specific fixes. Core ML is an *optional* attempt to offload the encoder to
the Apple Neural Engine / GPU for lower latency.

## Requirement
Core ML lowering calls Apple's `coremltools` native runtime, which **only exists on macOS**. You cannot
produce `laya_coreml.pte` on Linux — run `export_ios.sh --coreml` on a Mac.

## Known issue: gather index dtype
Laya's typed-decision head does a `gather` over the option-marker positions. ExecuTorch's Core ML
partitioner maps this to `gather_along_axis`, whose `indices` input must be **int32**, but the exported
graph feeds an fp32-expanded index tensor:

```
ValueError: Op "aten_gather_default" (op_type: gather_along_axis)
Input indices="aten_expand_copy_default_6" expects tensor or scalar of dtype from
type domain ['int32'] but got tensor[1,12,1024,fp32]
```

### Fix
In `model-src/rl_common.py`, where the head gathers marker hidden states, cast the index tensor to
`long`/`int32` **before** `expand`/`gather` (Core ML rejects a float index; XNNPACK tolerated it):

```python
# before:
idx = marker_pos.unsqueeze(-1).expand(-1, -1, hidden)          # may carry a float dtype
gathered = torch.gather(seq_out, 1, idx)

# after (Core ML-safe):
idx = marker_pos.long().unsqueeze(-1).expand(-1, -1, hidden)   # force integer indices
gathered = torch.gather(seq_out, 1, idx)
```

Re-run `./export_ios.sh --coreml`. If Core ML still refuses an op, it will simply partition that op back
to CPU — the model stays correct, just partly off the ANE. Always validate the Core ML `.pte` against
the Python reference (`make_testcases.py`) on-device before shipping it.

## Shipping the model on iOS
Put the chosen `.pte` in the app's **Documents** directory (or bundle it and copy on first launch), then
load it by relative name — the RN demo's `Platform.select()` already resolves `laya_int8.pte` on iOS.
