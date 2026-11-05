# Validation — 2026-11-05

Build: Anima LoRA Trainer 0.3.1
GUI baseline: suigin10/MOSS-LoRA-Trainer 74fe3171dfa1509bc8bb325e34f49b1b4d66dd1a
Backend target: kohya-ss/sd-scripts cb70b96858477e716a2096292ab8aab433277f38

## Verified

- Python syntax compilation for the distributed modules.
- 84 automated tests, including Anima schema/export checks, actual upstream schema/blueprint integration, cache conflict and invalidation, safetensors headers and explicit 2.9B rejection, archive traversal rejection, project storage/migration, resume counters, state retention, GUI project copy/save/reopen, JA/EN switching, equal separated Qwen3 button dimensions, folded Flow controls, scheduler combinations, extra argument parsing, and preservation of training hyperparameters by the runtime-speed button.
- CPU PyTorch / Accelerate optimizer-state restoration and scheduler-extension integration tests.
- Resume compatibility patch compiles against the actual pinned upstream train_network.py, with and without scheduler recalculation.
- Every exported training argument (including samples, compile, swap and resume options) exists in the pinned upstream argument definitions.
- GUI launched on Windows and visually inspected with the blue theme. Five setting tabs and one run/log tab; Flow controls are in tab 03's Advanced settings. Model details are folded by default.
- A small Fused AdamW backward/update executed successfully on the available RTX 4070 with PyTorch 2.7.0+cu128. This checks the fused optimizer primitive, not Anima training performance.
- All 14 offered optimizers completed 2 GPU backward/update steps through the actual pinned upstream optimizer factory. All 10 offered schedulers completed 20 GPU update/schedule steps. This used tiny 8x8 tensors and the existing local dependencies, not a full isolated installer environment. Results: docs/OPTIMIZER_SMOKE.json.
- New-project values were checked against the reference Anima-Standalone-Trainer GUI templates. Old projects retain explicit saved training values while obsolete no-op keys are removed on import.

## Not verified in this build session

- Download/install of the complete isolated training environment and all optional binary wheels.
- Full model loading, an end-to-end Anima training run, actual LoRA quality, sample generation, or downstream ComfyUI compatibility.
- End-to-end speed/VRAM comparisons, torch.compile performance, and the optional separate LLM Adapter weight overlay on a real model.

Model weights are not bundled. New defaults use LR 0.00005, 2000 steps, save every 100 steps; resolution is 1024px. No measured end-to-end speedup or VRAM capacity guarantee is claimed. Anima 2.9B / 40-layer training and SageAttention training are not supported.

## Reproduce

From the extracted application folder:

    python -m unittest discover -s tests -v

For the upstream schema integration tests, set ANIMA_TEST_SD_SCRIPTS to the
reviewed/installed sd-scripts source directory and run with voluptuous installed.
These tests execute the original schema/dataclass/blueprint definitions, without
loading model classes or tensors. Without that source path, three tests skip.

GUI tests require Windows/tkinter and an interactive desktop. Optional CPU optimizer tests skip when torch/accelerate are unavailable. The application itself starts without installing training packages.

## v0.3.0 Anima controls

- Tested module-LR conversion, zero-LR adapter removal, manual-regex precedence,
  regularization subset/loss export, dropout/norm options, old project migration,
  frozen Text Encoder / LLM Adapter enforcement, and GUI preset/save behavior.
- Executed the actual pinned networks/lora_anima.py against a tiny synthetic
  two-block model on RTX 4070: base LR, three component LRs, and each of the three
  component-disable modes completed two GPU updates. Original base weights stayed
  unchanged. No Text Encoder or LLM Adapter LoRAs were created. Results are in
  docs/ANIMA_LAYER_SMOKE.json. This does not replace full Anima training validation.
- GUI provenance sentences were removed. New configuration controls do not
  automatically apply the article example values or overwrite saved DiT-only runs.

## v0.3.1 regression

The user log showed successful environment installation, followed by
`extra keys not allowed @ data['datasets'][0]['prior_loss_weight']`.
The previous GUI put this setting in the dataset TOML. The upstream dataclass has
a prior_loss_weight member, but the input schema permits it only through argparse.
It is now written to the trainer TOML. The real ConfigSanitizer reproduces the old
failure and accepts the corrected export. BlueprintGenerator receives loss weights
0, 0.35, 1 and 2 through the trainer namespace and preserves the regularization flags.
This verifies config parsing/propagation, not an end-to-end training run.
