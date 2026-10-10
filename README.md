# MoVeR

**MoVeR: Cross-Partition Motion Scores and Relational Refinement for Tiny-UAV Event Segmentation**

This repository contains the public MoVeR full-model implementation for EV-UAV-format inference and evaluation. The ablation campaign, registry, training queue, and ablation artifacts are maintained separately in the private repository [MoVeR_ablation](https://github.com/bibocat12/MoVeR_ablation).

## Repository layout

```text
configs/
  mover_evuav.yaml             # MoVeR full-model configuration
dataset/
  ev_uav.py                    # official EV-UAV archive adapter
  mover_runtime.py             # preprocessing and full-model construction
models/
  model_vmc.py                 # verified motion context
  calibrated_mar.py            # context-calibrated relational refinement
  tramx_v7_sem.py              # full MoVeR encoder and prediction head
tools/
  train.py / test.py           # full-model train/evaluation entry points
train.py / test.py              # public-compatible entry points
```

The private repository contains the research-only ablation registry, campaign
launcher, per-arm aggregation, runtime campaign helpers, and experimental result
artifacts.

## Full-model configuration

The public entry points operate only on the complete MoVeR model. They preserve
one archive/clip per forward pass, select checkpoints from VAL IoU at
$\tau=0.90$, and keep TEST evaluation separate. Component switches and ablation
campaign tooling are not part of this public checkout; they are maintained in
[MoVeR_ablation](https://github.com/bibocat12/MoVeR_ablation).

```bash
python3 train.py \
  --config configs/mover_evuav.yaml \
  --device cuda:0 \
  --output checkpoints/mover

python3 test.py \
  --config configs/mover_evuav.yaml \
  --root /path/to/EV-UAV \
  --split test \
  --model-path checkpoints/mover/best.pt
```

Runtime measurement is model-only forward timing with CUDA synchronization;
preparation, file I/O, and host-to-device transfer are outside the timed region.

## Data format

The preferred layout is:

```text
EV-UAV/
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

Official archives contain `evs_norm`, `ev_loc`, and structured `ev`. The adapter uses `ev_loc` `(x,y,t)`, `evs_norm[:,:4]`, `ev['label']`, and `ev['name']`. The older flat cache layout is also accepted when files are named `train_*.npz`, `val_*.npz`, and `test_*.npz` and contain `locs`, `feats`, and `seg`.

## Runtime measurement

Runtime uses synchronized model-only forward timing. Preparation, file I/O, and
host-to-device transfer are outside the timed region.

```bash
python3 test.py \
  --config configs/mover_evuav.yaml \
  --root /path/to/EV-UAV \
  --split test \
  --model-path checkpoints/mover/best.pt \
  --measure-runtime
```

## Tests

```bash
python3 -m pytest -q
python3 -m compileall -q .
git diff --check
```


## Citation

```bibtex
@misc{mover,
  title={MoVeR: Cross-Partition Motion Scores and Relational Refinement for Tiny-UAV Event Segmentation},
  author={Le Gia Phuc},
  year={2026},
  url={https://github.com/bibocat12/MoVeR}
}
```
