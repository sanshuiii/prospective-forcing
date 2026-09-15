# Configuration reference

## Scientific parameters

Changing the following values creates a different experiment: initialization,
training prompts or embeddings, data order, random seed, denoising schedule,
number of optimizer-loop steps, batch size, rank count, sharding strategy,
loss weights, future horizons, candidate count, and M1 thresholds.

## Stage 1 model fields

- `prospective_enabled`: instantiate the future-feature draft.
- `prospective_draft_architecture`: `parallel_heads` in the released model.
- `prospective_num_heads`: attention heads in each draft module.
- `prospective_future_horizons`: two future latent chunks.
- `prospective_feature_weight`: normalized feature matching coefficient.
- `prospective_flow_weight`: auxiliary flow coefficient.
- `prospective_loss_weight`: total prospective objective coefficient.
- `prospective_anchor_policy`: eligible rolling anchor selection.

## Stage 2 M1 fields

- `m1_method`: `M1` enables two candidates and a learned verifier.
- `m1_candidate_seed`: deterministic candidate embedding initialization.
- `m1_epsilon_1`: reference normalized feature-error threshold.
- `m1_tau_run`: candidate execution threshold.
- `m1_tau_accept`: candidate acceptance threshold.
- `m1_loss_weight`: Stage 2 controller/verifier auxiliary coefficient.

## Execution-only overrides

Paths and inference GPU placement are environment variables listed in the main
README. These may change where work runs without changing the model objective.
Record the fully resolved YAML, package commit, dependency versions, GPU model,
and checkpoint hashes for every formal run.
