# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np

from models import model_utils


class CriticMLP(nn.Module):
    def __init__(self, obs_dim, cfg_network, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device
        self.dyn_recurrent = cfg_network["dyn_model_mlp"].get("recurrent", False)
        self.act_recurrent = cfg_network["actor_mlp"].get("recurrent", False)
        self.hid_act_to_crit = False
        self.hidden_to_value = cfg_network["dyn_model_mlp"].get("hidden_to_value", True)

        self.obs_total_size = obs_dim

        if self.dyn_recurrent and self.hidden_to_value:
            self.hidden_size = int(cfg_network["dyn_model_mlp"].get("hidden_size", 128))
            self.obs_total_size += self.hidden_size

        if self.act_recurrent and self.hid_act_to_crit:
            self.act_hidden_size = int(cfg_network["actor_mlp"].get("hidden_size", 128))
            self.obs_total_size += self.act_hidden_size

        self.layer_dims = [self.obs_total_size] + cfg_network['critic_mlp']['units'] + [1]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)

        self.obs_dim = obs_dim

    def forward(self, observations):
        return self.critic(observations)
