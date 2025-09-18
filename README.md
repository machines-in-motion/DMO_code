# First Order Model-Based RL through Decoupled Backpropagation (DMO) — Code

This repository accompanies the paper “First Order Model-Based RL through Decoupled Backpropagation,” which proposes Decoupled forward–backward Model-based policy Optimization (DMO): unroll trajectories in a high‑fidelity simulator while computing gradients via a learned differentiable model to enable efficient first‑order optimization without simulator derivatives.

## Project links
- Website: https://machines-in-motion.github.io/DMO/
- Paper (arXiv): [https://arxiv.org/abs/2509.00215v1](https://arxiv.org/abs/2509.00215v2)
- Announcement (X/Twitter): https://x.com/Jsphamigo/status/1964347428221415450

## Status
- Supported now: DMO‑SHAC on DFlex environments (training script and configs included).
- Coming soon: DMO‑SHAC on IsaacGym; DMO‑SAPO on DFlex, IsaacGym, and reawarped; active work‑in‑progress repository.

---

## Installation

### DFlex environments
```
conda env create -f diffrl_conda.yml
conda activate dmo_dflex
```
```
cd dflex
pip install -e .
```
The commands above create and activate the DMO environment for DFlex and install the local DFlex package in editable mode for development workflows.

---

## Quick start

### Train: DMO‑SHAC on DFlex
```
conda activate dmo_dflex
cd examples
python train_dmo_shac.py \
  --exp_name dmo_shac \
  --logdir ./logs/Ant/dmo_shac/20 \
  --cfg ./cfg/dmo_shac/ant.yaml \
  --seed 20
```
This launches DMO‑SHAC on Ant using the provided config, logging to the specified directory with a fixed seed for reproducibility.

---

## Configuration and logging
- Configs for DMO‑SHAC are under `examples/cfg/dmo_shac` and include task, optimizer, and logging settings.
- To disable Weights & Biases, set `wandb_track: False` in the config files in `examples/cfg/dmo_shac`; local logs still go under `--logdir`.
- Seeding is controlled via the `--seed` flag to facilitate reproducible experiments.
---

## Roadmap (WIP)
- Add cleaned scripts for DMO‑SHAC training on IsaacGym.
- Add cleaned scripts for DMO‑SAPO implementations on DFlex, IsaacGym, and reawarped.
---

## Citation
If this code or paper is useful, please cite the work below.
```
@inproceedings{amigo2025dmo,
  title     = {First Order Model-Based RL through Decoupled Backpropagation},
  author    = {Amigo, Joseph and Khorrambakht, Rooholla and Chane-Sane, Elliot and Mansard, Nicolas and Righetti, Ludovic},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2025},
  note      = {arXiv:2509.00215}
}
```

---
