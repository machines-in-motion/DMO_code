# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# gradient-based policy optimization by actor critic method
import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import argparse

import os
import sys
import yaml

import numpy as np
import copy

def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args

def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--cfg", "type": str, "default": "./cfg/dmo_shac/ant.yaml",
            "help": "Configuration file for training/playing"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights"},
        {"name": "--logdir", "type": str, "default": "logs/tmp/dmo_shac/"},
        {"name": "--save-interval", "type": int, "default": 0},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"},
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--render", "action": "store_true", "default": False,
            "help": "whether generate rendering file."},
        {"name": "--exp_name", "type": str, "default": "no_name",
            "help": "Experiment name for pairing with wandb and openrlbenchmark"},
        {"name": "--env_type", "type": str, "default": "dflex",
            "help": "isaac_gym or dflex for the simulator."},
        {"name": "--unroll-img", "action": "store_true", "default": False,
            "help": "whether to unroll the trajectories with the learned model instead of the simulator.."},
        {"name": "--multi-modal-cor", "action": "store_true", "default": False,
            "help": "whether to try to correct for the dynamics uncovered by the world model."},
        {"name": "--generate-dataset-only", "action": "store_true", "default": False,
            "help": "whether to be in pretraining dataset generation mode only (no training)."},
        {"name": "--load-pretrain-dataset", "action": "store_true", "default": False,
            "help": "whether to use an already generated pretraining dataset."},
        {"name": "--pretrain-dataset-path", "type": str, "default": "",
            "help": "path to the already generated pretraining dataset."},
        {"name": "--log-gradients", "action": "store_true", "default": False,
            "help": "wether to run the experiment comparing the gradients of the different methods."},
        {"name": "--no-stoch-dyn-model", "action": "store_true", "default": False,
            "help": "wether to use stochastic dynamics models or not."},
        {"name": "--no-residual-dyn-model", "action": "store_true", "default": False,
            "help": "wether to use residual dynamics models or not."},
        {"name": "--no-stoch-act-model", "action": "store_true", "default": False,
            "help": "wether to use a stochastic actor or not."},
        {"name": "--no-value", "action": "store_true", "default": False,
            "help": "wether to use a value function approximator or not."}]

    # parse arguments
    args = parse_arguments(
        description="DMO-SHAC",
        custom_parameters=custom_parameters)
    
    return args

if __name__ == '__main__':
    args = get_args()

    with open(args.cfg, 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    if args.play or args.test:
        cfg_train["params"]["config"]["num_actors"] = cfg_train["params"]["config"].get("player", {}).get("num_actors", 1)

    if args.env_type == "isaac_gym":
        import isaacgym

    from utils.common import get_time_stamp
    import torch
    import algorithms.dmo_shac as dmo_shac
    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    args.device = torch.device(args.device)
    vargs = vars(args)

    cfg_train["params"]["general"] = {}
    for key in vargs.keys():
        cfg_train["params"]["general"][key] = vargs[key]

    traj_optimizer = dmo_shac.DMOShac(cfg_train)

    if args.train:
        traj_optimizer.train()
    else:
        traj_optimizer.play(cfg_train)
