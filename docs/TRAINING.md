# Training

## Stage 1

`configs/training/stage1_cm_init.yaml` is the authoritative Stage 1 contract.
It initializes the causal student from `causal_cd.pt`, uses the Wan2.1 14B
real-score teacher, and trains for 3,000 optimizer-loop steps with eight ranks,
batch 1 per rank, full sharding, and seed 20260720. The denoising schedule is
`[1000, 800, 600, 400, 200]`.

The final file is normally
`runs/stage1/logs/checkpoint_model_003000/model.pt`. B0/A1 read the
`generator_ema` field from this file.

## Stage 2

`configs/training/stage2_m1.yaml` loads the Stage 1 `generator_ema`, attaches the
M1 K=2 controller/verifier, and runs 1,000 additional optimizer-loop steps.
Stage 2 continues the registered DMD generator/critic schedule while learning
the controller/verifier; it is not a controller-only freeze of the backbone.
The data order, batch, sharding, seed, denoising schedule, and base losses remain
aligned with Stage 1. M1 uses candidate seed 20260816, epsilon 0.12, routing
threshold 0.5, acceptance threshold 0.5, and auxiliary loss weight 0.1.

The final file is normally
`runs/stage2/logs/checkpoint_model_001000/model.pt`. M1 inference reads its
`generator_ema` field.

## Cached text conditions

Formal training expects precomputed BF16 UMT5 embeddings. The directory schema
is documented in `data/README.md`. `prepare_embeddings.py` creates exactly that
schema. Keep the prompt file, manifest, index, and all shards together.

## Distributed contract

The published launchers default to one node with eight visible GPUs. They also
support a two-node/four-GPU-per-node layout through the following variables:

```bash
export PF_NNODES=2
export PF_NPROC_PER_NODE=4
export PF_NODE_RANK=0              # use 1 on the second node
export PF_MASTER_ADDR=coordinator-host
export PF_MASTER_PORT=29500
bash scripts/training/train_stage1.sh
```

Both nodes run the same command with different node ranks. Preserve eight total
ranks, batch 1 per rank, global batch 8, and FSDP full sharding to match the
reference scientific configuration. Scheduler integration is intentionally
left to the user; no cluster-specific account, partition, or host is embedded.
