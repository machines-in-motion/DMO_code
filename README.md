# First Order Model-Based RL through Decoupled Backpropagation (DMO) — Code

This repository accompanies the paper “First Order Model-Based RL through Decoupled Backpropagation,” which proposes Decoupled forward–backward Model-based policy Optimization (DMO): unroll trajectories in a high‑fidelity simulator while computing gradients via a learned differentiable model to enable efficient first‑order optimization without simulator derivatives.

## Project links
- Website: https://machines-in-motion.github.io/DMO/
- Paper (arXiv): [https://arxiv.org/abs/2509.00215v1](https://arxiv.org/abs/2509.00215v2)
- Announcement (X/Twitter): https://x.com/Jsphamigo/status/1964347428221415450

## Status
- Supported now: DMO‑SHAC on DFlex environments (training script and configs included) and on IsaacGym Go2 quadrupedal walking env.
- Coming soon: DMO‑SHAC on IsaacGym for bipedal walking env; DMO‑SAPO on DFlex, IsaacGym, and reawarped; active work‑in‑progress repository.

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

### IsaacGym environments
Git clone our fork of IsaacGymEnvs:
```
git clone https://github.com/Jogima-cyber/IsaacGymEnvs.git
```
Download Isaac Gym: https://developer.nvidia.com/isaac-gym and run the following commands:
```
conda create -n IsaacGymEnvs python=3.8
conda activate IsaacGymEnvs

conda install pip
conda install -c conda-forge ninja

# Download Isaac Gym: https://developer.nvidia.com/isaac-gym 
cd path/to/isaacgym/python && pip install --no-cache-dir -e . 

# Install our fork of IsaacGymEnvs that you previously cloned
cd path/to/IsaacGymEnvs/ && pip install --no-cache-dir  -e .

pip uninstall torch tochvision
pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0
# Fix compability problem with numpy version
pip install --no-cache-dir "numpy<1.24"
pip install --no-cache-dir texttable av

# Add correct LD_LIBRARY_PATH on env start
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib --name $CONDA_DEFAULT_ENV
```

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

### Train: DMO‑SHAC on IsaacGym Go2 Quadrupedal Walking Env
```
conda activate IsaacGymEnvs
python train_dmo_shac.py \
  --exp_name dmo_shac \
  --logdir ./logs/Go2/dmo_shac/20 \
  --cfg ./cfg/dmo_shac/go2.yaml \
  --seed 20 \
  --env_type isaac_gym
cd examples
```

---

## Configuration and logging
- Configs for DMO‑SHAC are under `examples/cfg/dmo_shac` and include task, optimizer, and logging settings.
- To disable Weights & Biases, set `wandb_track: False` in the config files in `examples/cfg/dmo_shac`; local logs still go under `--logdir`.
- Seeding is controlled via the `--seed` flag to facilitate reproducible experiments.
---

## Roadmap (WIP)
- Add cleaned scripts for DMO‑SHAC training on IsaacGym for bipedal walking.
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
