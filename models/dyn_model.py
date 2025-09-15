# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

from models import model_utils

from torch.autograd import Function

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

class StochSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, obs_out_dim, cfg_network, learn_reward, device='cuda:0'):
        super().__init__()

        self.device = device

        self.recurrent = cfg_network["dyn_model_mlp"].get("recurrent", False)
        if self.recurrent:
            self.hidden_size = int(cfg_network["dyn_model_mlp"].get("hidden_size", 128))
            self.gru = nn.GRU(obs_dim + action_dim, self.hidden_size, batch_first=True).to(device)
            self.layer_dims = [obs_dim + action_dim + self.hidden_size] + cfg_network['dyn_model_mlp']['units'] + [obs_out_dim * 2]
        else:
            self.layer_dims = [obs_dim + action_dim] + cfg_network['dyn_model_mlp']['units'] + [obs_out_dim * 2]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                if i == 0:
                    modules.append(SimNorm(8))
                else:
                    modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1], eps=1e-03))

        self.dyn_model = nn.Sequential(*modules).to(device)

        self.learn_reward = learn_reward
        if self.learn_reward:
            self.num_bins = cfg_network['dyn_model_mlp']['num_bins']
            self.vmin = cfg_network['dyn_model_mlp']['vmin']
            self.vmax = cfg_network['dyn_model_mlp']['vmax']
            self.reward_layer_dims = [obs_dim + action_dim] + cfg_network['dyn_model_mlp']['reward_head_units'] + [self.num_bins]

            modules = []
            for i in range(len(self.reward_layer_dims) - 1):
                modules.append(nn.Linear(self.reward_layer_dims[i], self.reward_layer_dims[i + 1]))
                if i < len(self.reward_layer_dims) - 2:
                    modules.append(model_utils.get_activation_func(cfg_network['dyn_model_mlp']['activation']))
                    modules.append(torch.nn.LayerNorm(self.reward_layer_dims[i+1], eps=1e-03))

            self.reward_model = nn.Sequential(*modules).to(device)

        # Must be put outside in case of requires_grad=True and ensemble of models
        self.max_logvar = nn.Parameter(torch.ones(1, obs_out_dim) * 0.5, requires_grad=False).to(device)
        self.min_logvar = nn.Parameter(torch.ones(1, obs_out_dim) * -10, requires_grad=False).to(device)

        print(self.dyn_model)
        if self.learn_reward:
            print(self.reward_model)

    def forward(self, s, a, l = None):
        x = torch.cat([s, a], dim=-1)
        time_latent = l
        if self.recurrent:
            out, time_latent = self.gru(x, l)
            x = torch.cat([x, out], dim=-1)

        h = self.dyn_model(x)

        mean, logvar = h.chunk(2, dim=-1)
        # Differentiable clip, funny! A.1 in https://arxiv.org/pdf/1805.12114
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar, time_latent

    def reward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        rew = self.reward_model(x)
        return self.almost_two_hot_inv(rew)

    def raw_reward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        rew = self.reward_model(x)
        return rew

    def almost_two_hot_inv(self, x):
        """Converts a batch of soft two-hot encoded vectors to scalars."""
        if self.num_bins == 0 or self.num_bins == None:
            return x
        elif self.num_bins == 1:
            return symexp(x)
        # TODO this computation below can probably be optimized
        vals = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * vals, dim=-1, keepdim=True)
        return x

    def forward_with_log_ratio(self, s, a, next_obs_delta):
        x = torch.cat([s, a], dim=-1)

        h = self.dyn_model(x)

        mean, logvar = h.chunk(2, dim=-1)
        # Differentiable clip, funny! A.1 in https://arxiv.org/pdf/1805.12114
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        std = torch.sqrt(logvar.exp())
        dist = Normal(mean, std)

        return mean, logvar, dist.log_prob(next_obs_delta).sum(-1) - dist.log_prob(mean).sum(-1)

class RewardsFunction(Function):
    @staticmethod
    def forward(ctx, state, action, reward, model, obs_rms):
        # Save state, action, and model for backward
        ctx.save_for_backward(state, action)
        ctx.model = model
        ctx.obs_rms = obs_rms

        return reward

    @staticmethod
    def backward(ctx, grad_reward):
        state, action = ctx.saved_tensors
        model = ctx.model
        obs_rms = ctx.obs_rms

        # Compute the gradient of the dynamics model with respect to action and state
        with torch.enable_grad():
            reward_pred = model.reward(obs_rms.normalize(state), torch.tanh(action)).squeeze(-1) # get rid of tanh?

            if state.requires_grad:
                grad_state, = torch.autograd.grad(
                    reward_pred, state, grad_outputs=grad_reward, retain_graph=True
                )
            else:
                grad_state = None
            grad_action, = torch.autograd.grad(
                reward_pred, action, grad_outputs=grad_reward
            )

        # We only return gradients for state and action
        return grad_state, grad_action, None, None, None

class GradientSwapingFunction(Function):
    @staticmethod
    def forward(ctx, img_state, true_state):
        return true_state.clone()

    @staticmethod
    def backward(ctx, grad_true_next_state):
        return grad_true_next_state, None

