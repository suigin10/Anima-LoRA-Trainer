# Anima LoRA Trainer

Run Start Anima LoRA Trainer.bat with 64-bit Python 3.12 (3.11 also supported).
The GUI uses tkinter only. Use Easy Setup to install the isolated training backend.

1. Models: select Anima DiT, Qwen3-0.6B, and Qwen-Image VAE.
2. Dataset: add image folders, repeats, and matching .txt captions.
3. Training: review rank, alpha, learning rate, total steps, and save interval.
4. VRAM & speed: start with torch attention, checkpointing, and both caches enabled.
5. Run Preflight Check, then Start Training.

New-job defaults: rank 16 / alpha 16,
AdamW8bit, base LR 0.00005, weight decay 0.01, cosine, no warmup, 1024px, batch 1,
2000 steps, save every 100 steps, caption dropout 0.05 and 4 workers. Existing projects
retain their saved values. See DEFAULTS_SOURCE.md for the exact reference revision.

Only 28-block non-quantized Anima DiT LoRA is supported by the pinned backend.
Anima 2.9B (40 layers) is NOT supported. Kohaku, LoKr, LoHa and FP8 base training are excluded. SageAttention is not supported for training. Qwen3 is frozen.
LLM Adapter training is disabled too. This version trains DiT LoRA only.

T5 tokenizer files are bundled with sd-scripts. T5 model weights are unnecessary.
An external LLM Adapter is optional; otherwise use one embedded in the DiT.
Qwen3 can be a safetensors file or a complete Hugging Face model directory.

Cache files live in the project cache directory. Model, settings, image and caption
changes select a different cache path. Teacher images and their existing caches are
not modified. Old project caches may be removed manually while training is stopped.

TE caching cannot be combined with caption shuffle. Whole-caption dropout is supported.
Latent caching cannot be combined with color augmentation or random crop.
Block swap, CPU checkpoint offload and Unsloth checkpoint offload are mutually exclusive.

State resume restores optimizer/scheduler/RNG. Changing the total training limit
recalculates LR against that new total. Initial LoRA weights start a new optimizer.
Stopping does not guarantee a new state save: resume from the last complete state.

Triton is optional for ordinary training; torch.compile is off by default and requires
a compatible Triton/build environment. xFormers and Flash Attention are optional.

Use Optimize runtime only in tab 04 to enable caching, the 2D VAE, persistent workers,
cuDNN kernel selection, H2D-only swap (when swapping) and fused AdamW (AdamW only).
It preserves LR, rank, alpha, resolution, batch, steps and save intervals. It refuses
to change caches when augmentation or caption shuffle is enabled. It does not switch
your optimizer, attention backend or compile setting. Optional Triton installation
and GPU testing are available in Easy Setup. TF32 is off by default. Different GPU
kernels may introduce rounding differences. No end-to-end speedup has been measured.

LoRAs and states: data/projects/<ID>/runs/<run>/models/
Logs/configuration snapshots: the same run directory.
Backend: backend/anima-env/ (independent of the MOSS trainer environment).

Models: https://huggingface.co/circlestone-labs/Anima
Training: https://github.com/kohya-ss/sd-scripts
GUI derived from https://github.com/suigin10/MOSS-LoRA-Trainer (MIT).

The standalone Flow tab has been removed. Noise/flow and LoRA target controls now
live in the collapsed Advanced settings section in tab 03. Samples are tab 05.
The model file/folder buttons have equal sizes and a gap, and model details are
collapsed. Select either a single Qwen3 file or a complete HF folder, not both.

14 optimizers and 10 schedulers are available. Relevant scheduler detail fields
are enabled based on your selection. Extra arguments use one key=value per line.
ScheduleFree requires constant and warmup 0. The adafactor scheduler requires
Adafactor. Selecting an optimizer never rewrites the entered learning rate.
Changing the total limit on a state resume is unsupported for ScheduleFree and
AdafactorSchedule; use the same plan or initialize a new run from LoRA weights.

Anima component LRs are in tab 03 Advanced settings. Blank inherits the base LR;
0 disables that component; a positive number sets its LR. The comparison presets
change component LRs only. Detailed regex rates override positive component rates,
but cannot re-enable a zero-LR component. Relative-step Adafactor and ScheduleFree
reject positive component LRs rather than silently ignoring them.

Qwen3 Text Encoder, LLM Adapter, VAE and the original DiT weights stay frozen.
The shared sd-scripts flag network_train_unet_only means DiT LoRA only for Anima.
Turning off TE caching does not enable Text Encoder training. Old projects that
trained the LLM Adapter cannot resume here; create a new DiT-only project.

In tab 02, mark individual subsets as regularization images and set their loss
weight (default 1). At least one ordinary training subset is required. Neuron,
rank and module dropout and the LoRA weight norm limit default to 0 (disabled).
Regularization images and captions are prepared manually, not generated or edited
automatically. Real image training quality remains unverified.
