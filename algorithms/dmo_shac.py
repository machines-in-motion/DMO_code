# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from multiprocessing.sharedctypes import Value
import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
from tensorboardX import SummaryWriter
import yaml

from utils.common import *
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import utils.torch_utils as tu
from utils.running_mean_std import RunningMeanStd
from utils.dataset import CriticDataset, CriticAdvDataset
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
import models.actor
import models.critic
import models.dyn_model

from typing import NamedTuple
import gym
from gym.spaces import Box

def test_reorganized_array(init_arr, final_arr, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, name):
    """
    Test that the final rearranged array matches the expected indexing pattern of the original array.
    """

    # Basic shape and indexing checks for the first environment set
    assert init_arr[:dyn_rb_pos][::dyn_rb_num_envs].shape == final_arr[:new_num_envs * steps_per_env][::new_num_envs].shape, \
        f"[{name}] Shape mismatch after final reordering for the first set of environments."
    assert (init_arr[:dyn_rb_pos][::dyn_rb_num_envs] == final_arr[:new_num_envs * steps_per_env][::new_num_envs]).all(), \
        f"[{name}] Final indexing check failed for the first environment set."

    if new_num_envs >= 2:
        assert (init_arr[:dyn_rb_pos][1::dyn_rb_num_envs] == final_arr[:new_num_envs * steps_per_env][1::new_num_envs]).all(), \
            f"[{name}] Final indexing check failed for the first environment set (offset by 1)."

    # If we have more than one factor (env_factor >= 2), test the second environment set
    if env_factor >= 2:
        assert init_arr[:dyn_rb_pos][new_num_envs::dyn_rb_num_envs].shape == final_arr[new_num_envs * steps_per_env : 2 * new_num_envs * steps_per_env][::new_num_envs].shape, \
            f"[{name}] Shape mismatch after final reordering for the second environment factor set."
        assert (init_arr[:dyn_rb_pos][new_num_envs :: dyn_rb_num_envs] == final_arr[new_num_envs * steps_per_env : 2 * new_num_envs * steps_per_env][::new_num_envs]).all(), \
            f"[{name}] Indexing check failed for the second environment factor set."

        if new_num_envs >= 2:
            assert (init_arr[:dyn_rb_pos][new_num_envs + 1 :: dyn_rb_num_envs] == final_arr[new_num_envs * steps_per_env : 2 * new_num_envs * steps_per_env][1 :: new_num_envs]).all(), \
                f"[{name}] Indexing check failed for the second environment factor set (offset by 1)."

    assert (init_arr[:dyn_rb_pos][(env_factor - 1) * new_num_envs :: dyn_rb_num_envs] ==
            final_arr[(env_factor - 1) * new_num_envs * steps_per_env : env_factor * new_num_envs * steps_per_env][:: new_num_envs]).all(), \
        f"[{name}] Indexing check failed for the last environment factor set."
    if new_num_envs >= 2:
        # Test the last environment factor indexing
        assert (init_arr[:dyn_rb_pos][(env_factor - 1) * new_num_envs + 1 :: dyn_rb_num_envs] ==
                final_arr[(env_factor - 1) * new_num_envs * steps_per_env : env_factor * new_num_envs * steps_per_env][1 :: new_num_envs]).all(), \
            f"[{name}] Indexing check failed for the last environment factor set (offset by 1)."


def restructure_old_rb(dyn_rb, dyn_rb_observations, dyn_rb_actions, dyn_rb_next_observations,
                       dyn_rb_dones, dyn_rb_rewards, dyn_rb_p_ini_hidden_in, dyn_rb_num_envs,
                       dyn_rb_pos, new_num_envs):

    assert dyn_rb_pos < dyn_rb.buffer_size, "Assertion 1 failed: dyn_rb_pos must be less than buffer_size."
    assert new_num_envs <= dyn_rb_num_envs, "Assertion 2 failed: new_num_envs must be <= dyn_rb_num_envs."
    assert dyn_rb_num_envs % new_num_envs == 0, "Assertion 3 failed: dyn_rb_num_envs must be divisible by new_num_envs."
    assert dyn_rb_pos % new_num_envs == 0, "Assertion 4 failed: dyn_rb_pos must be divisible by new_num_envs."

    # Save initial copies of arrays for testing later
    init_obs = dyn_rb_observations.clone()
    init_next_obs = dyn_rb_next_observations.clone()
    init_act = dyn_rb_actions.clone()
    init_dones = dyn_rb_dones.clone()
    init_rewards = dyn_rb_rewards.clone()
    init_hidden = dyn_rb_p_ini_hidden_in.clone()

    # Compute parameters
    steps_per_env = dyn_rb_pos // dyn_rb_num_envs
    env_factor = dyn_rb_num_envs // new_num_envs

    obs_shape = dyn_rb_observations.shape[1:]
    act_shape = dyn_rb_actions.shape[1:]
    hidden_shape = dyn_rb_p_ini_hidden_in.shape[1:]

    # Reshape arrays before reorganizing
    dyn_rb_observations = dyn_rb_observations[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs, *obs_shape)
    dyn_rb_next_observations = dyn_rb_next_observations[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs, *obs_shape)
    dyn_rb_actions = dyn_rb_actions[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs, *act_shape)
    dyn_rb_dones = dyn_rb_dones[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs)
    dyn_rb_rewards = dyn_rb_rewards[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs)
    dyn_rb_p_ini_hidden_in = dyn_rb_p_ini_hidden_in[:dyn_rb_pos].view(steps_per_env, dyn_rb_num_envs, *hidden_shape)

    # Perform the reorganization
    # Observations
    dyn_rb_observations = dyn_rb_observations.reshape(steps_per_env, env_factor, new_num_envs, *obs_shape)
    dyn_rb_observations = dyn_rb_observations.swapaxes(0, 1).reshape(-1, *obs_shape)

    # Next Observations
    dyn_rb_next_observations = dyn_rb_next_observations.reshape(steps_per_env, env_factor, new_num_envs, *obs_shape)
    dyn_rb_next_observations = dyn_rb_next_observations.swapaxes(0, 1).reshape(-1, *obs_shape)

    # Actions
    dyn_rb_actions = dyn_rb_actions.reshape(steps_per_env, env_factor, new_num_envs, *act_shape)
    dyn_rb_actions = dyn_rb_actions.swapaxes(0, 1).reshape(-1, *act_shape)

    # Dones
    dyn_rb_dones = dyn_rb_dones.reshape(steps_per_env, env_factor, new_num_envs)
    dyn_rb_dones = dyn_rb_dones.swapaxes(0, 1).reshape(-1)

    # Rewards
    dyn_rb_rewards = dyn_rb_rewards.reshape(steps_per_env, env_factor, new_num_envs)
    dyn_rb_rewards = dyn_rb_rewards.swapaxes(0, 1).reshape(-1)

    # Hidden states
    dyn_rb_p_ini_hidden_in = dyn_rb_p_ini_hidden_in.reshape(steps_per_env, env_factor, new_num_envs, *hidden_shape)
    dyn_rb_p_ini_hidden_in = dyn_rb_p_ini_hidden_in.swapaxes(0, 1).reshape(-1, *hidden_shape)

    # Call the test function for each array
    test_reorganized_array(init_obs, dyn_rb_observations, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Observations")
    test_reorganized_array(init_next_obs, dyn_rb_next_observations, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Next Observations")
    test_reorganized_array(init_act, dyn_rb_actions, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Actions")
    test_reorganized_array(init_dones, dyn_rb_dones, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Dones")
    test_reorganized_array(init_rewards, dyn_rb_rewards, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Rewards")
    test_reorganized_array(init_hidden, dyn_rb_p_ini_hidden_in, dyn_rb_pos, dyn_rb_num_envs, new_num_envs, steps_per_env, env_factor, "Hidden States")

    dyn_rb.observations[:dyn_rb_pos] = dyn_rb_observations
    dyn_rb.next_observations[:dyn_rb_pos] = dyn_rb_next_observations
    dyn_rb.actions[:dyn_rb_pos] = dyn_rb_actions
    dyn_rb.dones[:dyn_rb_pos] = dyn_rb_dones
    dyn_rb.rewards[:dyn_rb_pos] = dyn_rb_rewards
    dyn_rb.p_ini_hidden_in[:dyn_rb_pos] = dyn_rb_p_ini_hidden_in

    dyn_rb.started_adding = True
    dyn_rb.pos = dyn_rb_pos
    dyn_rb.prev_start_idx = dyn_rb_pos - new_num_envs
    dyn_rb.prev_stop_idx = dyn_rb_pos
    dyn_rb.markers[dyn_rb.prev_start_idx : dyn_rb.prev_stop_idx] = 1
    dyn_rb.prev_overflow = False
    dyn_rb.prev_overflow_size = 0

class SeqReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    p_ini_hidden_in: torch.Tensor
    p_ini_hidden_out: torch.Tensor
    mask: torch.Tensor

class SeqReplayBuffer():
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        max_episode_length,
        seq_len,
        num_envs,
        hidden_size,
        critic_rnn = False,
        storing_device = "cpu",
        training_device = "cpu",
        handle_timeout_termination = True,
    ):
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length
        self.seq_len = seq_len
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.hidden_size = hidden_size

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.critic_rnn = critic_rnn
        self.storing_device = storing_device
        self.training_device = training_device

        self.observations = torch.zeros((self.buffer_size, *self.observation_space), dtype=torch.float32, device = storing_device)
        self.next_observations = torch.zeros((self.buffer_size, *self.observation_space), dtype=torch.float32, device = storing_device)

        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32, device = storing_device)
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32, device = storing_device)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.bool, device = storing_device)
        self.p_ini_hidden_in = torch.zeros((self.buffer_size, 1, self.hidden_size), dtype=torch.float32, device = storing_device)

        # For the current episodes that started being added to the replay buffer
        # but aren't done yet. We want to still sample from them, however the masking
        # needs a termination point to not overlap to the next episode when full or even to the empty
        # part of the buffer when not full.
        self.markers = torch.zeros((self.buffer_size,), dtype=torch.bool, device = storing_device)
        self.started_adding = False

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros((self.buffer_size,), dtype=torch.float32, device = storing_device)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        p_ini_hidden_in,
        truncateds = None
    ):
        start_idx = self.pos
        stop_idx = min(self.pos + obs.shape[0], self.buffer_size)
        b_max_idx = stop_idx - start_idx

        overflow = False
        overflow_size = 0
        if self.pos + obs.shape[0] > self.buffer_size:
            overflow = True
            overflow_size = self.pos + obs.shape[0] - self.buffer_size

        assert start_idx % self.num_envs == 0, f"start_idx is not a multiple of {self.num_envs}"
        assert stop_idx % self.num_envs == 0, f"stop_idx is not a multiple of {self.num_envs}"
        assert b_max_idx == 0 or b_max_idx == self.num_envs, f"b_max_idx is not either 0 or {self.num_envs}"

        # Copy to avoid modification by reference
        self.observations[start_idx : stop_idx] = obs[: b_max_idx].clone().to(self.storing_device)

        self.next_observations[start_idx : stop_idx] = next_obs[: b_max_idx].clone().to(self.storing_device)

        self.actions[start_idx : stop_idx] = action[: b_max_idx].clone().to(self.storing_device)
        self.rewards[start_idx : stop_idx] = reward[: b_max_idx].clone().to(self.storing_device)
        self.dones[start_idx : stop_idx] = done[: b_max_idx].clone().to(self.storing_device)
        self.p_ini_hidden_in[start_idx : stop_idx] = p_ini_hidden_in.swapaxes(0, 1)[: b_max_idx].clone().to(self.storing_device)

        # Current episodes last transition marker
        self.markers[start_idx : stop_idx] = 1
        # We need to unmark previous transitions as last
        # but only if it is not the first add to the replay buffer
        if self.started_adding:
            self.markers[self.prev_start_idx : self.prev_stop_idx] = 0
            if self.prev_overflow:
                self.markers[: self.prev_overflow_size] = 0
        self.started_adding = True
        self.prev_start_idx = start_idx
        self.prev_stop_idx = stop_idx
        self.prev_overflow = overflow
        self.prev_overflow_size = overflow_size

        if self.handle_timeout_termination:
            self.timeouts[start_idx : stop_idx] = truncateds[: b_max_idx].clone().to(self.storing_device)

        assert overflow_size == 0 or overflow_size == self.num_envs, f"overflow_size is not either 0 or {self.num_envs}"
        if overflow:
            self.full = True
            self.observations[: overflow_size] = obs[b_max_idx :].clone().to(self.storing_device)

            self.next_observations[: overflow_size] = next_obs[b_max_idx :].clone().to(self.storing_device)

            self.actions[: overflow_size] = action[b_max_idx :].clone().to(self.storing_device)
            self.rewards[: overflow_size] = reward[b_max_idx :].clone().to(self.storing_device)
            self.dones[: overflow_size] = done[b_max_idx :].clone().to(self.storing_device)
            self.p_ini_hidden_in[: overflow_size] = p_ini_hidden_in.swapaxes(0, 1)[b_max_idx :].clone().to(self.storing_device)

            # Current episodes last transition marker
            self.markers[: overflow_size] = 1
            if self.handle_timeout_termination:
                self.timeouts[: overflow_size] = truncateds[b_max_idx :].clone().to(self.storing_device)
            self.pos = overflow_size
        else:
            self.pos += obs.shape[0]

    def sample(self, batch_size) -> SeqReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper_bound, size = (batch_size,), device = self.storing_device)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds) -> SeqReplayBufferSamples:
        # Using modular arithmetic we get the indices of all the transitions of the episode starting from batch_inds
        # we get "episodes" of length self.seq_len, but their true length may be less, they can have ended before that
        # we'll deal with that using a mask
        # Using flat indexing we can actually slice through a tensor using
        # different starting points for each dimension of an axis
        # as long as the slice size remains constant
        # [1, 2, 3].repeat_interleave(3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        # [1, 2, 3].repeat(3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
        all_indices_flat = (batch_inds.repeat_interleave(self.seq_len) + torch.arange(self.seq_len, device = self.storing_device).repeat(batch_inds.shape[0]) * self.num_envs) % self.buffer_size
        #all_indices_next_flat = (batch_inds.repeat_interleave(self.seq_len) + torch.arange(1, self.seq_len + 1, device = self.device).repeat(batch_inds.shape[0]) * self.num_envs) % self.buffer_size
        gathered_obs = self.observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.observations.shape[1:]))
        gathered_next_obs = self.next_observations[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.observations.shape[1:]))

        gathered_actions = self.actions[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len, *self.actions.shape[1:]))
        gathered_dones = self.dones[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        gathered_truncateds = self.timeouts[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        gathered_rewards = self.rewards[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))

        gathered_p_ini_hidden_in = self.p_ini_hidden_in[batch_inds].swapaxes(0, 1)
        gathered_p_ini_hidden_out = self.p_ini_hidden_in[(batch_inds + self.num_envs) % self.buffer_size].swapaxes(0, 1)

        gathered_markers = self.markers[all_indices_flat].reshape((batch_inds.shape[0], self.seq_len))
        mask = torch.cat([
            torch.ones((batch_inds.shape[0], 1), device = self.storing_device),
            (1 - (gathered_dones | gathered_markers).float()).cumprod(dim = 1)[:, 1:]
        ], dim = 1)
        data = (
            gathered_obs.to(self.training_device),
            gathered_actions.to(self.training_device),
            gathered_next_obs.to(self.training_device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (gathered_dones.float() * (1 - gathered_truncateds)).to(self.training_device),
            gathered_rewards.to(self.training_device),
            gathered_p_ini_hidden_in.to(self.training_device),
            gathered_p_ini_hidden_out.to(self.training_device),
            mask.to(self.training_device),
        )
        return SeqReplayBufferSamples(*data)

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer():
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        storing_device = "cpu",
        training_device = "cpu",
        n_envs = 1,
        optimize_memory_usage = False,
        handle_timeout_termination = True,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.storing_device = storing_device
        self.training_device = training_device
        self.n_envs = n_envs

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device = storing_device)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device = storing_device)

        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device = storing_device)

        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        truncateds = None
    ):
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.clone().to(self.storing_device)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = next_obs.clone().to(self.storing_device)

        self.actions[self.pos] = action.clone().to(self.storing_device)
        self.rewards[self.pos] = reward.clone().to(self.storing_device)
        self.dones[self.pos] = done.clone().to(self.storing_device)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = truncateds.to(self.storing_device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size, env = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds, env=env)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :].to(self.training_device),
            self.actions[batch_inds, env_indices, :].to(self.training_device),
            next_obs.to(self.training_device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1).to(self.training_device),
            self.rewards[batch_inds, env_indices].reshape(-1, 1).to(self.training_device),
        )
        return ReplayBufferSamples(*data)

# Taken from https://github.com/nicklashansen/tdmpc2/blob/5f6fadec0fec78304b4b53e8171d348b58cac486/tdmpc2/common/math.py#L5C1-L9C47
def soft_ce(pred, target, num_bins, vmin, vmax):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, num_bins, vmin, vmax)
    return -(target * pred).sum(-1, keepdim=True)

def two_hot(x, num_bins, vmin, vmax):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_idx = torch.floor((x - vmin) / bin_size).long()
    bin_offset = ((x - vmin) / bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot

def gather_all_gradients(model):
    # Ensure all gradients are available
    gradients = []
    for param in model.parameters():
        if param.grad is not None:  # Some parameters might not have gradients
            gradients.append(param.grad.view(-1))  # Flatten the gradient tensor

    # Concatenate all gradients into a single tensor
    mega_gradient = torch.cat(gradients)
    return mega_gradient

def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

class DMOShac:
    def __init__(self, cfg):
        self.env_type = cfg["params"]["general"]["env_type"]

        seeding(cfg["params"]["general"]["seed"])
        self.log_gradients = cfg["params"]["general"]['log_gradients']
        self.no_stoch_dyn_model = cfg["params"]["general"]['no_stoch_dyn_model'] or cfg["params"]["config"].get("no_stoch_dyn_model", False)
        self.no_residual_dyn_model = cfg["params"]["general"]['no_residual_dyn_model']
        self.no_stoch_act_model = cfg["params"]["general"]['no_stoch_act_model'] or cfg["params"]["config"].get("no_stoch_act_model", False)
        self.no_value = cfg["params"]["general"]['no_value']

        if self.env_type == "dflex":
            import dflex as df
            import envs
            env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
            no_grad = True
            if self.log_gradients:
                no_grad = False

            self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                              device = cfg["params"]["general"]["device"], \
                              render = cfg["params"]["general"]["render"], \
                              seed = cfg["params"]["general"]["seed"], \
                              episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                              stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                              MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                              no_grad = no_grad)
            self.max_episode_length = self.env.episode_length
        elif self.env_type == "isaac_gym":
            import isaacgym
            import isaacgymenvs
            self.env = isaacgymenvs.make(
                seed=cfg["params"]["general"]["seed"],
                task=cfg["params"]["isaac_gym"]["name"],
                num_envs=cfg["params"]["config"]["num_actors"],
                sim_device="cuda:0",
                rl_device="cuda:0",
                headless=True
            )
            self.env.num_actions = self.env.num_acts
            self.max_episode_length = self.env.max_episode_length
            self.num_dyn_obs = self.env.num_dyn_obs
        else:
            raise ValueError(
                f"env type {self.env_type} is not supported."
            )

        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_obs = ', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.num_act_obs = self.num_obs
        if hasattr(self.env, "num_act_obs"):
            self.num_act_obs = self.env.num_act_obs

        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg['params']['config'].get('gamma', 0.99)

        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step') # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.steps_num_schedule = cfg["params"]["config"].get("steps_num_schedule", False)
        if self.steps_num_schedule:
            self.steps_num_start = int(cfg["params"]["config"]["steps_num_start"])

        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.dyn_model_lr = float(cfg['params']['config']['dyn_model_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        self.actor_lr_schedule_min = float(cfg["params"]["config"].get("actor_lr_schedule_min", 1e-5))
        self.train_value_function = cfg["params"]["config"].get("train_value_function", True) and not self.no_value

        self.decouple_value_actors = cfg["params"]["config"].get("decouple_value_actors", False)
        if self.decouple_value_actors:
            self.apg_num_actors = cfg["params"]["config"].get("apg_num_actors", 256)

        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)

        self.obs_rms = None
        self.act_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            if self.env_type == "dflex":
                self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            else:
                self.obs_rms = RunningMeanStd(shape = (self.num_dyn_obs), device = self.device)

        if cfg['params']['config'].get('act_rms', False):
            self.act_rms = RunningMeanStd(shape = (self.num_actions), device = self.device)

        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]

        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            if cfg['params']['config'].get('wandb_track', False):
                import wandb

                cfg['env_id'] = cfg['params']['diff_env']['name']
                cfg['exp_name'] = cfg["params"]["general"]['exp_name']

                wandb.init(
                    project=cfg['params']['config']['wandb_project_name'],
                    sync_tensorboard=True,
                    config=cfg,
                    name=self.name,
                    monitor_gym=True,
                    save_code=True,
                )

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (cfg['params']['config']['player'].get('determenistic', False) or cfg['params']['config']['player'].get('deterministic', False))
            self.steps_num = self.env.episode_length

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP') # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = cfg["params"]["network"].get("critic", 'CriticMLP')

        self.multi_modal_cor = cfg['params']['general']['multi_modal_cor']
        if self.multi_modal_cor:
            self.dyn_model_name = "StochSSMCor"
        else:
            self.dyn_model_name = cfg["params"]["network"].get("dyn_model", 'StochSSM')
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor = actor_fn(self.num_act_obs, self.num_actions, cfg['params']['network'], device = self.device)
        critic_fn = getattr(models.critic, self.critic_name)

        self.dyn_recurrent = cfg["params"]["network"]["dyn_model_mlp"].get("recurrent", False)
        if self.dyn_recurrent:
            self.dyn_seq_len = int(cfg["params"]["network"]["dyn_model_mlp"].get("seq_len", 50))
            self.dyn_hidden_size = int(cfg["params"]["network"]["dyn_model_mlp"].get("hidden_size", 128))
            self.hidden_to_value = cfg["params"]["network"]["dyn_model_mlp"].get("hidden_to_value", True)

        self.separate = cfg['params']['network']['critic_mlp'].get("separate", True)
        self.asymetrical_value = cfg["params"]["network"]["critic_mlp"].get("asymetrical", True)
        if self.separate:
            if self.env_type == "dflex":
                self.critic = critic_fn(self.num_obs, cfg['params']['network'], device = self.device)
            else:
                if self.asymetrical_value:
                    self.critic = critic_fn(self.num_dyn_obs, cfg['params']['network'], device = self.device)
                else:
                    self.critic = critic_fn(self.num_act_obs, cfg['params']['network'], device = self.device)
        else:
            self.critic = self.actor

        self.learn_reward = cfg["params"]["config"].get("learn_reward", False)
        if self.learn_reward:
            self.num_bins = cfg['params']['network']['dyn_model_mlp']['num_bins']
            self.vmin = cfg['params']['network']['dyn_model_mlp']['vmin']
            self.vmax = cfg['params']['network']['dyn_model_mlp']['vmax']

        self.dyn_model_load = cfg["params"]["config"].get("dyn_model_load", False)
        self.load_pretrain_dataset = cfg['params']['general']['load_pretrain_dataset']
        if self.load_pretrain_dataset:
           self.actor, dyn_rb_observations, dyn_rb_actions, dyn_rb_next_observations, dyn_rb_dones, dyn_rb_rewards, dyn_rb_p_ini_hidden_in, dyn_rb_num_envs, dyn_rb_pos, self.obs_rms, self.act_rms, self.ret_rms = torch.load(cfg["params"]["general"]["pretrain_dataset_path"])

        if self.dyn_model_load:
            _, _, _, self.dyn_model, _, _, _ = torch.load(cfg["params"]["config"]["dyn_model_checkpoint"])
            self.actor, _, _, _, self.obs_rms, _, _ = torch.load(cfg["params"]["config"]["act_model_checkpoint"])
            self.dyn_loss = 0.
        else:
            dyn_model_fn = getattr(models.dyn_model, self.dyn_model_name)
            if self.env_type == "dflex":
                self.dyn_model = dyn_model_fn(self.num_obs, self.num_actions, self.num_obs, cfg['params']['network'], self.learn_reward, device = self.device)
            else:
                self.dyn_model = dyn_model_fn(self.num_dyn_obs, self.num_actions, self.num_dyn_obs, cfg['params']['network'], self.learn_reward, device = self.device)

        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.dyn_model.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        if cfg['params']['general']['train']:
            self.save('init_policy')

        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = cfg['params']['config']['betas'], lr = self.critic_lr)
        self.dyn_model_optimizer = torch.optim.Adam(self.dyn_model.parameters(), lr = self.dyn_model_lr)

        # replay buffer
        self.imagined_batch_size = int(cfg["params"]["config"].get("imagined_batch_size", 0))

        self.act_recurrent = cfg["params"]["network"]["actor_mlp"].get("recurrent", False)
        if self.act_recurrent:
            self.act_hidden_size = int(cfg["params"]["network"]["actor_mlp"].get("hidden_size", 128))

        self.obs_total_size = 0

        if self.env_type == "dflex":
            self.obs_total_size += self.num_obs
        else:
            if self.separate and self.asymetrical_value:
                self.obs_total_size += self.num_dyn_obs
            else:
                self.obs_total_size += self.num_act_obs

        if self.dyn_recurrent and self.hidden_to_value:
            self.obs_total_size += self.dyn_hidden_size

        if self.act_recurrent:
            self.obs_total_size += self.act_hidden_size

        value_num_envs = self.num_envs
        if self.decouple_value_actors:
            value_num_envs = self.apg_num_actors

        self.obs_buf = torch.zeros((self.steps_num, value_num_envs + self.imagined_batch_size, self.obs_total_size), dtype = torch.float32, device = self.device)

        self.rew_buf = torch.zeros((self.steps_num, value_num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, value_num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, value_num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, value_num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((value_num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)

        # real replay buffer for dyn model
        self.dyn_buffer_size = int(float(cfg["params"]["config"].get("dyn_buffer_size", 5e6)))
        rb_num_envs = self.num_envs
        if self.decouple_value_actors:
            rb_num_envs = self.apg_num_actors

        self.generate_dataset_only = cfg['params']['general']['generate_dataset_only']
        if self.env_type == "dflex":
            if self.dyn_recurrent:
                self.dyn_rb = SeqReplayBuffer(
                    self.dyn_buffer_size,
                    (self.num_obs,),
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                    self.max_episode_length,
                    self.dyn_seq_len,
                    rb_num_envs,
                    self.dyn_hidden_size,
                    storing_device = "cpu",
                    training_device = self.device,
                )
            else:
                self.dyn_rb = ReplayBuffer(
                    self.dyn_buffer_size,
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_obs])),
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                    storing_device = "cpu",
                    training_device = self.device,
                    n_envs = rb_num_envs,
                )
        else:
            if self.dyn_recurrent:
                self.dyn_rb = SeqReplayBuffer(
                    self.dyn_buffer_size,
                    (self.num_dyn_obs,),
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                    self.max_episode_length,
                    self.dyn_seq_len,
                    rb_num_envs,
                    self.dyn_hidden_size,
                    storing_device = self.device, # "cpu",
                    training_device = self.device,
                )
            else:
                self.dyn_rb = ReplayBuffer(
                    self.dyn_buffer_size,
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_dyn_obs])),
                    Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                    storing_device = "cpu",
                    training_device = self.device,
                    n_envs = rb_num_envs,
                )

        if self.load_pretrain_dataset and self.dyn_recurrent:
            restructure_old_rb(self.dyn_rb, dyn_rb_observations, dyn_rb_actions, dyn_rb_next_observations, dyn_rb_dones, dyn_rb_rewards, dyn_rb_p_ini_hidden_in, dyn_rb_num_envs, dyn_rb_pos, self.num_envs)
        elif self.load_pretrain_dataset and not self.dyn_recurrent:
            raise NotImplementedError("Loading a pretraining dataset when not using RNNs in the dyn model hasn't been implemented yet.")

        self.dyn_udt = int(cfg["params"]["config"].get("dyn_udt", 256))
        self.init_dyn_udt = int(cfg["params"]["config"].get("init_dyn_udt", 1000))
        self.min_replay = int(cfg["params"]["config"].get("min_replay", 4000))
        self.dyn_pred_batch_size = int(cfg["params"]["config"].get("dyn_pred_batch_size", 1024))
        self.pretrained = False
        self.filter_sigma_events = cfg["params"]["config"].get("filter_sigma_events", False)
        self.unroll_img = cfg["params"]["general"]['unroll_img']

        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

    def fill_replay_buffer(self, deterministic = False):
        filled_nb = 0
        self.p_hidden_in = None
        if self.log_gradients:
            self.p_hidden_in_ = None
        if self.dyn_recurrent:
            self.p_hidden_in = torch.zeros((1, self.num_envs, self.dyn_hidden_size), device=self.device)

        self.act_hidden_in = None
        if self.act_recurrent:
            self.act_hidden_in = torch.zeros((1, self.num_envs, self.act_hidden_size), device=self.device)
        self.last_action_recorded = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.env_type == "dflex":
                obs = self.env.initialize_trajectory()
            else:
                obs = self.env.dyn_obs_buf.clone()

            if self.generate_dataset_only:
                with torch.no_grad():
                    dummy_actions, _ = self.actor(self.env.full2partial_state(obs), deterministic = deterministic, l = self.act_hidden_in)
                    dummy_actions = torch.tanh(dummy_actions)
                    actions_min, _ = dummy_actions.min(dim = 0)
                    actions_max, _ = dummy_actions.max(dim = 0)

            raw_obs = obs.clone()
            if self.obs_rms is not None:
                # update obs rms
                if not self.dyn_model_load or True:
                    with torch.no_grad():
                        self.obs_rms.update(obs)
                # normalize the current obs
                obs = self.obs_rms.normalize(obs)
            while filled_nb < self.min_replay:
                if self.env_type == "dflex":
                    if self.no_stoch_act_model:
                        actions, self.act_hidden_in = self.actor(obs, deterministic = True, l = self.act_hidden_in)
                    else:
                        actions, self.act_hidden_in = self.actor(obs, deterministic = deterministic, l = self.act_hidden_in)
                    actions = torch.tanh(actions)
                    _, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), actions.unsqueeze(-2), self.p_hidden_in) # to get the hiddden for the replay buffer

                    if self.generate_dataset_only:
                        cur_min, _ = actions.min(dim = 0)
                        cur_max, _ = actions.max(dim = 0)
                        actions_min = torch.min(actions_min, cur_min)
                        actions_max = torch.max(actions_max, cur_max)

                    self.last_last_action_recorded = self.last_action_recorded.clone()
                    self.last_action_recorded = actions.clone()
                    next_obs, rew, done, extra_info = self.env.neurodiff_step(actions)
                    real_next_obs = next_obs.clone()
                    if done.any():
                        done_idx = torch.argwhere(done).squeeze()
                        real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
                        self.last_action_recorded[done_idx] = 0.
                        self.last_last_action_recorded[done_idx] = 0.
                else:
                    if self.no_stoch_act_model:
                        actions, self.act_hidden_in = self.actor(self.env.full2partial_state(obs), deterministic = True, l = self.act_hidden_in)
                    else:
                        actions, self.act_hidden_in = self.actor(self.env.full2partial_state(obs), deterministic = deterministic, l = self.act_hidden_in)
                    actions = torch.tanh(actions)
                    if hasattr(self.env, "action_scale_prime"):
                        actions = actions * self.env.action_scale_prime + self.env.action_bias
                    self.last_last_action_recorded = self.last_action_recorded.clone()
                    self.last_action_recorded = actions.clone()

                    self.time_report.start_timer("env step")
                    next_obs, rew, done, extra_info = self.env.step(actions)
                    self.time_report.end_timer("env step")

                    done = extra_info['dones']
                    next_obs = self.env.dyn_obs_buf.clone()
                    real_next_obs = next_obs.clone()
                    if done.any():
                        done_idx = torch.argwhere(done).squeeze()
                        real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
                        self.last_action_recorded[done_idx] = 0.
                        self.last_last_action_recorded[done_idx] = 0.
                        if self.act_recurrent:
                            self.act_hidden_in[:, done_idx] = 0.

                if self.act_rms is not None:
                    # update act rms
                    with torch.no_grad():
                        self.act_rms.update(actions)

                if self.decouple_value_actors:
                    if self.dyn_recurrent:
                        self.dyn_rb.add(
                            raw_obs.detach()[:self.apg_num_actors],
                            real_next_obs.detach()[:self.apg_num_actors],
                            actions.detach()[:self.apg_num_actors],
                            rew.detach()[:self.apg_num_actors],
                            done.float()[:self.apg_num_actors],
                            torch.zeros((1, self.apg_num_actors, self.dyn_hidden_size), device=self.device),
                            done.float()[:self.apg_num_actors]
                        )
                    else:
                        self.dyn_rb.add(
                            raw_obs.detach()[:self.apg_num_actors],
                            real_next_obs.detach()[:self.apg_num_actors],
                            actions.detach()[:self.apg_num_actors],
                            rew.detach()[:self.apg_num_actors],
                            done.float()[:self.apg_num_actors],
                            done.float()[:self.apg_num_actors]
                        )
                else:
                    if self.dyn_recurrent:
                        self.dyn_rb.add(raw_obs.detach(), real_next_obs.detach(), actions.detach(), rew.detach(), done.float(), self.p_hidden_in, done.float())
                    else:
                        self.dyn_rb.add(raw_obs.detach(), real_next_obs.detach(), actions.detach(), rew.detach(), done.float(), done.float())

                if self.decouple_value_actors:
                    filled_nb += self.apg_num_actors
                else:
                    filled_nb += self.num_envs

                raw_obs = next_obs.clone()
                if self.obs_rms is not None:
                    # update obs rms
                    if not self.dyn_model_load or True:
                        with torch.no_grad():
                            self.obs_rms.update(next_obs.clone())
                    # normalize the current obs
                    obs = self.obs_rms.normalize(next_obs.clone())

        if self.log_gradients:
            if self.dyn_recurrent:
                self.p_hidden_in_ = self.p_hidden_in.clone()

        if self.generate_dataset_only:
            filename = "ppo_actions_minmax.pt"
            print("actions_min", actions_min.cpu())
            print("actions_max", actions_max.cpu())
            torch.save((actions_min.cpu(), actions_max.cpu()), os.path.join(self.log_dir, "{}.pt".format(filename)))
        print("Filling done")

    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs + self.imagined_batch_size, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device) # /!\ not compatible with steps num schedule

        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)
        if self.log_gradients:
            rew_acc_ = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
            next_values_ = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
            actor_loss_ = torch.tensor(0., dtype = torch.float32, device = self.device)

            rew_acc_diff = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
            next_values_diff = torch.zeros((self.steps_num + 1, self.num_envs + self.imagined_batch_size), dtype = torch.float32, device = self.device)
            actor_loss_diff = torch.tensor(0., dtype = torch.float32, device = self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.act_rms is not None:
                act_rms = copy.deepcopy(self.act_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        if self.env_type == "dflex":
            obs = self.env.initialize_trajectory()
            if self.log_gradients:
                obs_diff = obs.clone()
                obs_ = obs.clone()
        else:
            obs = self.env.dyn_obs_buf.clone()

        if self.imagined_batch_size:
            data = self.dyn_rb.sample(self.imagined_batch_size)
            obs = torch.cat([obs, data.observations], dim = 0)

        raw_obs = obs.clone()
        if self.log_gradients:
            raw_obs_ = obs.clone()
            raw_obs_diff = obs.clone()

        if self.unroll_img:
            true_raw_obs = obs.clone()
        if self.obs_rms is not None:
            # update obs rms
            if not self.dyn_model_load or True:
                with torch.no_grad():
                    if self.unroll_img:
                        self.obs_rms.update(true_raw_obs)
                    else:
                        self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)
            if self.log_gradients:
                obs_ = obs_rms.normalize(obs_)
                obs_diff = obs_rms.normalize(obs_diff)

        if self.dyn_recurrent:
            self.p_hidden_in = self.p_hidden_in.detach()
            if self.log_gradients:
                self.p_hidden_in_ = self.p_hidden_in_.detach()

        if self.act_recurrent:
            self.act_hidden_in = self.act_hidden_in.detach()

        last_actions = torch.zeros((self.steps_num + 1, self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        last_last_actions = torch.zeros((self.steps_num + 2, self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        last_actions[0] = self.last_action_recorded.clone()
        last_last_actions[0] = self.last_last_action_recorded.clone()
        last_last_actions[1] = self.last_action_recorded.clone()

        horizon = self.steps_num
        if self.steps_num_schedule: # /!\ Not compatible with value function learning yet
            horizon = int((self.steps_num - self.steps_num_start) * (self.iter_count / self.max_epochs) + self.steps_num_start)

        for i in range(horizon):
            # collect data for critic training
            with torch.no_grad():
                if self.unroll_img and False: # We tried with and it decreases significantly results
                    obs_for_obs_buf = obs_rms.normalize(true_raw_obs.clone())
                else:
                    obs_for_obs_buf = obs.clone()
                if not self.separate or not self.asymetrical_value:
                    obs_for_obs_buf = self.env.full2partial_state(obs_for_obs_buf)
                if self.dyn_recurrent and self.hidden_to_value:
                    obs_for_obs_buf = torch.cat([obs_for_obs_buf, self.p_hidden_in.clone().squeeze(0)], dim=-1)
                if self.act_recurrent:
                    obs_for_obs_buf = torch.cat([obs_for_obs_buf, self.act_hidden_in.clone().squeeze(0)], dim=-1)
                if self.decouple_value_actors:
                    self.obs_buf[i] = obs_for_obs_buf[:self.apg_num_actors]
                else:
                    self.obs_buf[i] = obs_for_obs_buf

            if self.env_type == "dflex":
                if self.no_stoch_act_model:
                    actions, self.act_hidden_in = self.actor(obs, deterministic = True, l = self.act_hidden_in)
                else:
                    actions, self.act_hidden_in = self.actor(obs, deterministic = deterministic, l = self.act_hidden_in)
                if self.log_gradients:
                    actions_, _ = self.actor(obs_, deterministic = deterministic, l = None)
                    actions_diff, _ = self.actor(obs_diff, deterministic = deterministic, l = None)
            else:
                if self.no_stoch_act_model:
                    actions, self.act_hidden_in = self.actor(self.env.full2partial_state(obs.clone()), deterministic = True, l = self.act_hidden_in)
                else:
                    actions, self.act_hidden_in = self.actor(self.env.full2partial_state(obs.clone()), deterministic = deterministic, l = self.act_hidden_in)

            actions = torch.tanh(actions)
            if self.log_gradients:
                actions_ = torch.tanh(actions_)
                actions_diff = torch.tanh(actions_diff)

            if hasattr(self.env, "action_scale_prime"):
                actions = actions * self.env.action_scale_prime + self.env.action_bias
            if self.act_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.act_rms.update(actions)
            last_last_actions[i + 2] = actions.clone()
            last_actions[i + 1] = actions.clone()
            self.last_last_action_recorded = self.last_action_recorded.clone()
            self.last_action_recorded = actions.detach().clone()

            if self.unroll_img:
                if self.no_residual_dyn_model:
                     raise NotImplementedError("The no-residual-dyn-model flag is not compatible with the unroll-img flag.")
                if self.env_type == "dflex":
                    next_obs, rew, done, extra_info = self.env.neurodiff_step(actions.detach())
                else:
                    next_obs, rew, done, extra_info = self.env.step(actions.detach())
                    next_obs = self.env.dyn_obs_buf.clone()

                if self.dyn_recurrent:
                    real_next_obs, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), actions.unsqueeze(-2), self.p_hidden_in)
                else:
                    if self.act_rms is not None:
                        real_next_obs, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), act_rms.normalize(actions).unsqueeze(-2), self.p_hidden_in)
                    else:
                        real_next_obs, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), actions.unsqueeze(-2), self.p_hidden_in)

                real_next_obs = real_next_obs.squeeze(-2) + raw_obs

                if(done.any()):
                    done_idx = torch.argwhere(done).squeeze()
                    last_actions[i + 1, done_idx] = 0.
                    self.last_action_recorded[done_idx] = 0.
                    last_last_actions[i + 1, done_idx] = 0.
                    last_last_actions[i + 2, done_idx] = 0.
                    self.last_last_action_recorded[done_idx] = 0.
                    if self.act_recurrent:
                        self.act_hidden_in[:, done_idx] = 0.
            else:
                if self.multi_modal_cor:
                    real_next_obs, next_obs, rew, done, extra_info = models.dyn_model.DynamicsFunctionCor.apply(raw_obs, actions, self.dyn_model, self.env, obs_rms)
                else:
                    if self.env_type == "dflex":
                        if self.log_gradients:
                            next_obs_diff, rew_diff, done, extra_info = self.env.neurodiff_step(actions_diff[:self.num_envs])
                            next_obs = next_obs_diff.detach().clone()
                            rew = rew_diff.detach().clone()
                        else:
                            with torch.no_grad():
                                next_obs, rew, done, extra_info = self.env.neurodiff_step(actions[:self.num_envs].detach())

                        # Here we must wire the real next obs into the backprop graph, which next_obs isn't in case of last obs of the trajectory
                        # because next_obs is the obs after the env was reset
                        real_next_obs = next_obs.clone()
                        if self.log_gradients:
                            real_next_obs_diff = next_obs_diff.clone()
                    elif self.env_type == "isaac_gym":
                        with torch.no_grad():
                            self.time_report.start_timer("env step")
                            next_obs, rew, done, extra_info = self.env.step(actions[:self.num_envs].detach())
                            self.time_report.end_timer("env step")
                        done = extra_info['dones']
                        next_obs = self.env.dyn_obs_buf.clone()
                        real_next_obs = next_obs.clone()

                    if(done.any()):
                        done_idx = torch.argwhere(done).squeeze()
                        real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx].detach()

                        if self.log_gradients:
                            real_next_obs_diff[done_idx] = extra_info['obs_before_reset'][done_idx].clone()

                        last_actions[i + 1, done_idx] = 0.
                        self.last_action_recorded[done_idx] = 0.
                        last_last_actions[i + 1, done_idx] = 0.
                        last_last_actions[i + 2, done_idx] = 0.
                        self.last_last_action_recorded[done_idx] = 0.
                        if self.act_recurrent:
                            self.act_hidden_in[:, done_idx] = 0.

                    if self.imagined_batch_size:
                        if self.dyn_recurrent:
                            raise NotImplementedError("This function is not yet implemented.")
                        else:
                            actions[self.num_envs:] = actions[self.num_envs:].detach()
                            img_real_next_obs, _, _ = self.dyn_model(obs[self.num_envs:].unsqueeze(-2), actions[self.num_envs:].unsqueeze(-2), self.p_hidden_in)
                            img_real_next_obs = img_real_next_obs.squeeze(-2) + raw_obs[self.num_envs:]
                            real_next_obs = torch.cat([real_next_obs, img_real_next_obs], dim = 0)
                            img_done = self.env.imgDone(img_real_next_obs)
                            img_next_obs = img_real_next_obs.clone()
                            new_data = self.dyn_rb.sample(int(img_done.sum().item()))
                            img_done_env_ids = img_done.nonzero(as_tuple = False).squeeze(-1)
                            img_next_obs[img_done_env_ids] = new_data.observations
                            next_obs = torch.cat([next_obs, img_next_obs], dim = 0)
                            done = torch.cat([done, img_done], dim = 0)

                    if self.dyn_recurrent:
                        img_next_obs_delta, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), actions.unsqueeze(-2), self.p_hidden_in)
                    else:
                        if self.act_rms is not None:
                            img_next_obs_delta, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), act_rms.normalize(actions).unsqueeze(-2), self.p_hidden_in)
                        else:
                            img_next_obs_delta, _, self.p_hidden_in = self.dyn_model(obs.unsqueeze(-2), actions.unsqueeze(-2), self.p_hidden_in)

                    if self.no_residual_dyn_model:
                        img_next_obs = img_next_obs_delta.squeeze(-2)
                    else:
                        img_next_obs = img_next_obs_delta.squeeze(-2) + raw_obs
                    real_next_obs = models.dyn_model.GradientSwapingFunction.apply(img_next_obs, real_next_obs.clone())

            replay_addable = True

            if self.log_gradients:
                if self.unroll_img:
                     raise NotImplementedError("The unroll-img flag is not compatible with the log-gradients flag.")
                if self.no_residual_dyn_model:
                     raise NotImplementedError("The no-residual-dyn-model flag is not compatible with the log-gradients flag.")
                if self.env_type != "dflex":
                     raise NotImplementedError("The log-gradients flag is only compatible with the dflex environement.")

                if self.dyn_recurrent:
                    real_next_obs_, _, self.p_hidden_in_ = self.dyn_model(obs_.unsqueeze(-2), actions_.unsqueeze(-2), self.p_hidden_in_)
                else:
                    if self.act_rms is not None:
                        real_next_obs_, _, self.p_hidden_in_ = self.dyn_model(obs_.unsqueeze(-2), act_rms.normalize(actions_).unsqueeze(-2), self.p_hidden_in_)
                    else:
                        real_next_obs_, _, self.p_hidden_in_ = self.dyn_model(obs_.unsqueeze(-2), actions_.unsqueeze(-2), self.p_hidden_in_)
                real_next_obs_ = real_next_obs_.squeeze(-2) + raw_obs_

            if self.filter_sigma_events and self.env_type == "dflex":
                if (~torch.isfinite(next_obs)).sum() > 0:
                    print_warning("Got inf next_obs from sim")
                    nan_idx = torch.any(~torch.isfinite(next_obs), dim=-1)
                    next_obs[nan_idx] = 0.0
                    if self.log_gradients:
                        next_obs_diff[nan_idx] = 0.0
                    replay_addable = False

                if (~torch.isfinite(real_next_obs)).sum() > 0:
                    print_warning("Got inf real_next_obs from sim")
                    nan_idx = torch.any(~torch.isfinite(real_next_obs), dim=-1)
                    real_next_obs[nan_idx] = 0.0
                    if self.log_gradients:
                        real_next_obs_diff[nan_idx] = 0.0
                    replay_addable = False

                nan_idx = torch.any(real_next_obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large real_next_obs from sim")
                    real_next_obs[nan_idx] = 0.0
                    if self.log_gradients:
                        real_next_obs_diff[nan_idx] = 0.0
                    replay_addable = False

                nan_idx = torch.any(next_obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large next_obs from sim")
                    next_obs[nan_idx] = 0.0
                    if self.log_gradients:
                        next_obs_diff[nan_idx] = 0.0
                    replay_addable = False

                alpha = 6
                below_alphasigmas_masks = torch.any(real_next_obs < (-alpha * torch.sqrt(obs_rms.var + 1e-5) + obs_rms.mean).unsqueeze(0), dim=-1)
                above_alphasigmas_masks = torch.any(real_next_obs > (alpha * torch.sqrt(obs_rms.var + 1e-5) + obs_rms.mean).unsqueeze(0), dim=-1)
                alphasigmas_masks = torch.logical_or(below_alphasigmas_masks, above_alphasigmas_masks)
                if alphasigmas_masks.any():
                    print_warning("Got real_next_obs not in alpha sigma interval from sim")
                    replay_addable = False

            if replay_addable:
                if self.unroll_img:
                    true_real_next_obs = next_obs.clone()
                    if(done.any()):
                        done_idx = torch.argwhere(done).squeeze()
                        true_real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
                    if self.dyn_recurrent:
                        self.dyn_rb.add(true_raw_obs[:self.num_envs].detach(), true_real_next_obs[:self.num_envs].detach(), actions[:self.num_envs].detach(), rew[:self.num_envs].detach(), done[:self.num_envs].float(), self.p_hidden_in[:, :self.num_envs].detach(), done[:self.num_envs].float())
                    else:
                        self.dyn_rb.add(true_raw_obs[:self.num_envs].detach(), true_real_next_obs[:self.num_envs].detach(), actions[:self.num_envs].detach(), rew[:self.num_envs].detach(), done[:self.num_envs].float(), done[:self.num_envs].float())
                else:
                    if self.decouple_value_actors:
                        if self.dyn_recurrent:
                            self.dyn_rb.add(
                                raw_obs[:self.apg_num_actors].detach(),
                                real_next_obs[:self.apg_num_actors].detach(),
                                actions[:self.apg_num_actors].detach(),
                                rew[:self.apg_num_actors].detach(),
                                done[:self.apg_num_actors].float(),
                                self.p_hidden_in[:, :self.apg_num_actors].detach(),
                                done[:self.apg_num_actors].float()
                            )
                        else:
                            self.dyn_rb.add(
                                raw_obs[:self.apg_num_actors].detach(),
                                real_next_obs[:self.apg_num_actors].detach(),
                                actions[:self.apg_num_actors].detach(),
                                rew[:self.apg_num_actors].detach(),
                                done[:self.apg_num_actors].float(),
                                done[:self.apg_num_actors].float()
                            )
                    else:
                        if self.dyn_recurrent:
                            self.dyn_rb.add(raw_obs[:self.num_envs].detach(), real_next_obs[:self.num_envs].detach(), actions[:self.num_envs].detach(), rew[:self.num_envs].detach(), done[:self.num_envs].float(), self.p_hidden_in[:, :self.num_envs].detach(), done[:self.num_envs].float())
                        else:
                            self.dyn_rb.add(raw_obs[:self.num_envs].detach(), real_next_obs[:self.num_envs].detach(), actions[:self.num_envs].detach(), rew[:self.num_envs].detach(), done[:self.num_envs].float(), done[:self.num_envs].float())

            ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
            if self.filter_sigma_events:
                def create_hook():
                    def hook(grad):
                        torch.nan_to_num(grad, 0.0, 0.0, 0.0, out = grad)
                    return hook

                if real_next_obs.requires_grad:
                    real_next_obs.register_hook(create_hook())
                if actions.requires_grad:
                    actions.register_hook(create_hook())

                if self.log_gradients:
                    if real_next_obs_.requires_grad:
                        real_next_obs_.register_hook(create_hook())
                    if actions_.requires_grad:
                        actions_.register_hook(create_hook())

                    if real_next_obs_diff.requires_grad:
                        real_next_obs_diff.register_hook(create_hook())
                    if actions_diff.requires_grad:
                        actions_diff.register_hook(create_hook())
            #################################################

            #with torch.no_grad(): #/!\#
            #    raw_rew = rew.clone() #/!\#

            # Differential reward recomputation
            self.time_report.start_timer("recomputing reward")
            if self.learn_reward:
                recalculated_rew = models.dyn_model.RewardsFunction.apply(raw_obs, actions, rew.clone(), self.dyn_model, obs_rms)
            else:
                recalculated_rew = self.env.diffRecalculateReward(real_next_obs, actions, last_actions[i], last_last_actions[i], imagined_trajs = self.imagined_batch_size)
                """if not torch.allclose(rew[:self.num_envs], recalculated_rew[:self.num_envs], rtol=1e-05, atol=1e-06, equal_nan=False) and not self.unroll_img: # and self.env_type == "dflex":
                    print(i, (rew[:self.num_envs] != recalculated_rew[:self.num_envs]), rew[:self.num_envs], recalculated_rew[:self.num_envs], (rew[:self.num_envs] - recalculated_rew[:self.num_envs]))
                    print((rew[:self.num_envs] != recalculated_rew[:self.num_envs]).sum(), done.sum(), (done == (rew[:self.num_envs] != recalculated_rew[:self.num_envs])).sum())
                    print((~torch.isclose(rew[:self.num_envs], recalculated_rew[:self.num_envs], rtol=1e-05, atol=1e-07, equal_nan=False)).sum())
                    print((rew[:self.num_envs] != recalculated_rew[:self.num_envs]).argwhere())
                    print('recalculated reward error')

                    raise ValueError"""
            rew = recalculated_rew.clone()

            with torch.no_grad(): #/!\#
                raw_rew = rew.clone() #/!\#

            if self.log_gradients:
                recalculated_rew_ = self.env.diffRecalculateReward(real_next_obs_, actions_, last_actions[i], last_last_actions[i], imagined_trajs = self.imagined_batch_size)
                rew_ = recalculated_rew_.clone()

            self.time_report.end_timer("recomputing reward")

            # Next obs recomputation, in case of reset, we must put back the state after reset into obs
            prev_obs = obs.clone()
            obs = real_next_obs.clone()
            if self.log_gradients:
                obs_ = real_next_obs_.clone()
                obs_diff = real_next_obs_diff.clone()

            if done.any():
                done_idx = torch.argwhere(done).squeeze()
                obs[done_idx] = next_obs[done_idx].detach() # Cut off from the graph
                if self.log_gradients:
                    obs_[done_idx] = next_obs[done_idx].detach()
                    obs_diff[done_idx] = next_obs_diff[done_idx].detach()

                if self.dyn_recurrent:
                    self.p_hidden_in[:, done_idx] = torch.zeros((1, self.num_envs, self.dyn_hidden_size), device=self.device)[:, done_idx]
                    if self.log_gradients:
                        self.p_hidden_in_[:, done_idx] = torch.zeros((1, self.num_envs, self.dyn_hidden_size), device=self.device)[:, done_idx]

            ########################

            # scale the reward
            rew = rew * self.rew_scale
            if self.log_gradients:
                rew_ = rew_ * self.rew_scale
                rew_diff = rew_diff * self.rew_scale

            raw_obs = obs.clone()
            if self.log_gradients:
                raw_obs_ = obs_.clone()
                raw_obs_diff = obs_diff.clone()

            if self.unroll_img:
                true_raw_obs = next_obs.clone()
            if self.obs_rms is not None:
                # update obs rms
                if not self.dyn_model_load or True:
                    with torch.no_grad():
                        if self.unroll_img:
                            self.obs_rms.update(true_raw_obs)
                        else:
                            self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)
                if self.log_gradients:
                    obs_ = obs_rms.normalize(obs_)
                    obs_diff = obs_rms.normalize(obs_diff)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)

                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            if self.train_value_function:
                value_obs = obs
                if self.log_gradients:
                    value_obs_ = obs_
                    value_obs_diff = obs_diff
                if not self.separate or not self.asymetrical_value:
                    value_obs = self.env.full2partial_state(value_obs)
                if self.dyn_recurrent and self.hidden_to_value:
                    value_obs = torch.cat([value_obs, self.p_hidden_in.clone().squeeze(0)], dim=-1)
                    if self.log_gradients:
                        value_obs_ = torch.cat([value_obs_, self.p_hidden_in_.clone().squeeze(0)], dim=-1)
                        value_obs_diff = torch.cat([value_obs_diff, self.p_hidden_in_diff.clone().squeeze(0)], dim=-1)

                if self.act_recurrent:
                    value_obs = torch.cat([value_obs, self.act_hidden_in.clone().squeeze(0)], dim=-1)

                if self.separate:
                    next_values[i + 1] = self.target_critic(value_obs).squeeze(-1)
                    if self.log_gradients:
                        next_values_[i + 1] = self.target_critic(value_obs_).squeeze(-1)
                        next_values_diff[i + 1] = self.target_critic(value_obs_diff).squeeze(-1)
                else:
                    next_values[i + 1] = self.target_critic.value(value_obs).squeeze(-1)

                self.time_report.start_timer("handling done for value function")
                for id in done_env_ids:
                    if self.env_type == "dflex" and id < self.num_envs and (torch.isnan(extra_info['obs_before_reset'][id]).sum() > 0 \
                        or torch.isinf(extra_info['obs_before_reset'][id]).sum() > 0 \
                        or (torch.abs(extra_info['obs_before_reset'][id]) > 1e6).sum() > 0): # ugly fix for nan values
                        next_values[i + 1, id] = 0.
                        if self.log_gradients:
                            next_values_[i + 1, id] = 0.
                            next_values_diff[i + 1, id] = 0.
                    elif id >= self.num_envs or self.episode_length[id] < self.max_episode_length: # early termination
                        next_values[i + 1, id] = 0.
                        if self.log_gradients:
                            next_values_[i + 1, id] = 0.
                            next_values_diff[i + 1, id] = 0.
                    else: # otherwise, use terminal value critic to estimate the long-term performance
                        if self.obs_rms is not None:
                            real_obs = obs_rms.normalize(real_next_obs[id])
                            if self.log_gradients:
                                real_obs_ = obs_rms.normalize(real_next_obs_[id])
                                real_obs_diff = obs_rms.normalize(real_next_obs_diff[id])
                        else:
                            real_obs = real_next_obs[id]
                        value_real_obs = real_obs.clone()
                        if self.log_gradients:
                            value_real_obs_ = real_obs_.clone()
                            value_real_obs_diff = real_obs_diff.clone()

                        if not self.separate or not self.asymetrical_value:
                            value_real_obs = self.env.full2partial_state(value_real_obs)
                        if self.dyn_recurrent and self.hidden_to_value:
                            value_real_obs = torch.cat([value_real_obs, self.p_hidden_in.clone().squeeze(0)[id]], dim=-1)
                            if self.log_gradients:
                                value_real_obs_ = torch.cat([value_real_obs_, self.p_hidden_in_.clone().squeeze(0)[id]], dim=-1)
                                value_real_obs_diff = torch.cat([value_real_obs_diff, self.p_hidden_in_diff.clone().squeeze(0)[id]], dim=-1)

                        if self.act_recurrent:
                            value_real_obs = torch.cat([value_real_obs, self.act_hidden_in.clone().squeeze(0)[id]], dim=-1)
                        if self.separate:
                            next_values[i + 1, id] = self.target_critic(value_real_obs).squeeze(-1)
                            if self.log_gradients:
                                next_values_[i + 1, id] = self.target_critic(value_real_obs_).squeeze(-1)
                                next_values_diff[i + 1, id] = self.target_critic(value_real_obs_diff).squeeze(-1)
                        else:
                            next_values[i + 1, id] = self.target_critic.value(value_real_obs).squeeze(-1)
                self.time_report.end_timer("handling done for value function")

                if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                    print('next value error')
                    raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew
            if self.log_gradients:
                rew_acc_[i + 1, :] = rew_acc_[i, :] + gamma * rew_
                rew_acc_diff[i + 1, :] = rew_acc_diff[i, :] + gamma * rew_diff

            if i < horizon - 1:
                if self.train_value_function:
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
                    if self.log_gradients:
                        actor_loss_ = actor_loss_ + (- rew_acc_[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values_[i + 1, done_env_ids]).sum()
                        actor_loss_diff = actor_loss_diff + (- rew_acc_diff[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values_diff[i + 1, done_env_ids]).sum()
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids]).sum() #- self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                # terminate all envs at the end of optimization iteration
                if self.train_value_function:
                    actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
                    if self.log_gradients:
                        actor_loss_ = actor_loss_ + (- rew_acc_[i + 1, :] - self.gamma * gamma * next_values_[i + 1, :]).sum()
                        actor_loss_diff = actor_loss_diff + (- rew_acc_diff[i + 1, :] - self.gamma * gamma * next_values_diff[i + 1, :]).sum()
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, :]).sum() #- self.gamma * gamma * next_values[i + 1, :]).sum()

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.
            if self.log_gradients:
                rew_acc_[i + 1, done_env_ids] = 0.
                rew_acc_diff[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                if self.decouple_value_actors:
                    self.rew_buf[i] = rew[:self.apg_num_actors].clone()
                    if i < horizon - 1:
                        self.done_mask[i] = done[:self.apg_num_actors].clone().to(torch.float32)
                    else:
                        self.done_mask[i, :] = 1.
                    self.next_values[i] = next_values[i + 1, :self.apg_num_actors].clone()
                else:
                    self.rew_buf[i] = rew.clone()
                    if i < horizon - 1:
                        self.done_mask[i] = done.clone().to(torch.float32)
                    else:
                        self.done_mask[i, :] = 1.
                    self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            self.time_report.start_timer("collecting episode loss")
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                done_env_ids = done_env_ids[done_env_ids < self.num_envs]
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                            print('ep loss error')
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.
                        self.episode_discounted_loss[done_env_id] = 0.
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.
            self.time_report.end_timer("collecting episode loss")

        actor_loss /= horizon * (self.num_envs + self.imagined_batch_size)
        if self.log_gradients:
            actor_loss_ /= horizon * (self.num_envs + self.imagined_batch_size)
            actor_loss_diff /= horizon * (self.num_envs + self.imagined_batch_size)

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()

        self.step_count += horizon * self.num_envs

        if self.log_gradients:
            return actor_loss, actor_loss_, actor_loss_diff

        return actor_loss

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic = False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = int)
        episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)

        obs = self.env.reset()
        self.actor.eval()

        games_cnt = 0
        steps_cnt = 0
        while games_cnt < num_games:
        #while steps_cnt < 1000:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            actions, _ = self.actor(obs, deterministic = deterministic)

            obs, rew, done, _ = self.env.step(torch.tanh(actions))

            episode_length += 1
            steps_cnt += 1

            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.
                    episode_discounted_loss[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.
                    games_cnt += 1

        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            value_num_envs = self.num_envs
            if self.decouple_value_actors:
                value_num_envs = self.apg_num_actors
            Ai = torch.zeros(value_num_envs + self.imagined_batch_size, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(value_num_envs + self.imagined_batch_size, dtype = torch.float32, device = self.device)
            lam = torch.ones(value_num_envs + self.imagined_batch_size, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        if self.separate:
            predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        else:
            predicted_values = self.critic.value(batch_sample['obs']).squeeze(-1)

        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        if self.env_type == "dflex":
            self.env.clear_grad()
            self.env.reset()
        #else:
        #    self.env.launch()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games = num_games, deterministic = not self.stochastic_evaluation)
        print_info('mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss, mean_policy_discounted_loss, mean_episode_length))

    def train_dyn_model(self):
        train_steps = self.dyn_udt
        if not self.pretrained:
            train_steps = self.init_dyn_udt
            print("pre training")

        log_total_dyn_loss = 0.0
        log_total_dyn_stoch_loss = 0.0
        log_total_reward_loss = 0.0

        if self.pretrained:
            data = self.dyn_rb.sample(self.dyn_pred_batch_size * train_steps)
            perm_indices = torch.randperm(self.dyn_pred_batch_size * train_steps)
        for ts in range(train_steps):
            if not self.pretrained:
                data = self.dyn_rb.sample(self.dyn_pred_batch_size)
                observations_concat = data.observations
                actions_concat = data.actions
                next_observations_concat = data.next_observations
                if self.learn_reward:
                    reward_concat = data.rewards
                if self.dyn_recurrent:
                    p_ini_hidden_in_concat = data.p_ini_hidden_in
                    mask_concat = data.mask
            else:
                cur_perm_indices = perm_indices[ts * self.dyn_pred_batch_size : (ts + 1) * self.dyn_pred_batch_size]
                observations_concat = data.observations[cur_perm_indices]
                actions_concat = data.actions[cur_perm_indices]
                next_observations_concat = data.next_observations[cur_perm_indices]
                if self.learn_reward:
                    reward_concat = data.rewards[cur_perm_indices]
                if self.dyn_recurrent:
                    p_ini_hidden_in_concat = data.p_ini_hidden_in[:, cur_perm_indices]
                    mask_concat = data.mask[cur_perm_indices]

            if self.obs_rms is not None:
                # normalize the current obs
                state = self.obs_rms.normalize(observations_concat.clone())

            if self.act_rms is not None:
                actions_concat = self.act_rms.normalize(actions_concat.clone())

            if self.no_residual_dyn_model:
                target_observations = next_observations_concat
            else:
                target_observations = next_observations_concat - observations_concat

            if self.multi_modal_cor:
                latent = self.dyn_model.encode(state, actions_concat, next_state)
                pred = self.dyn_model(state, actions_concat, latent)
            else:
                if self.dyn_recurrent:
                    pred = self.dyn_model(state, actions_concat, p_ini_hidden_in_concat)
                else:
                    pred = self.dyn_model(state, actions_concat)

            pred_next_logvar = pred[1]
            pred_next_obs = pred[0]
            if self.learn_reward:
                pred_reward = self.dyn_model.reward(state, actions_concat)

            if self.dyn_recurrent:
                inv_var = torch.exp(-pred_next_logvar)
                mse_loss_inv = ((torch.pow(pred_next_obs - target_observations, 2) * inv_var) * mask_concat.unsqueeze(-1)).sum() / (mask_concat.sum() * self.num_obs)
                var_loss = (pred_next_logvar * mask_concat.unsqueeze(-1)).sum() / (mask_concat.sum() * self.num_obs)
                if self.no_stoch_dyn_model:
                    loss = (F.mse_loss(pred_next_obs, target_observations, reduction = "none") * mask_concat.unsqueeze(-1)).sum() / (mask_concat.sum() * self.num_obs)
                else:
                    loss = mse_loss_inv + var_loss
            else:
                inv_var = torch.exp(-pred_next_logvar)
                mse_loss_inv = (torch.pow(pred_next_obs - target_observations, 2) * inv_var).mean()
                var_loss = pred_next_logvar.mean()
                if self.no_stoch_dyn_model:
                    loss = F.mse_loss(pred_next_obs, target_observations, reduction = "none").mean()
                else:
                    loss = mse_loss_inv + var_loss

            if self.learn_reward:
                if self.num_bins == 0:
                    reward_loss = F.mse_loss(pred_reward, reward_concat)
                    log_total_reward_loss += reward_loss.item()
                    loss += reward_loss
                else:
                    reward_loss = soft_ce(pred_reward, reward_concat, self.num_bins, self.vmin, self.vmax).mean()
                    log_total_reward_loss += reward_loss.item()
                    loss += reward_loss

            self.dyn_model_optimizer.zero_grad()
            loss.backward()
            self.dyn_model_optimizer.step()

            with torch.no_grad():
                log_total_dyn_stoch_loss += loss.item()

                if self.dyn_recurrent:
                    loss = (F.mse_loss(pred_next_obs, target_observations, reduction = "none") * mask_concat.unsqueeze(-1)).sum() / (mask_concat.sum() * self.num_obs)
                else:
                    loss = F.mse_loss(pred_next_obs, target_observations, reduction = "none").mean()
                log_total_dyn_loss += loss.item()

        self.writer.add_scalar("scalars/dyn_loss", log_total_dyn_loss / ts, self.iter_count)
        self.writer.add_scalar("scalars/reward_loss", log_total_reward_loss / ts, self.iter_count)
        self.writer.add_scalar("scalars/dyn_stoch_loss", log_total_dyn_stoch_loss / ts, self.iter_count)
        self.dyn_loss = log_total_dyn_loss / ts
        if not self.pretrained:
            print("pre training end")
            self.pretrained = True

    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.add_timer("dyn model training")
        self.time_report.add_timer("env step")

        self.time_report.add_timer("collecting episode loss")
        self.time_report.add_timer("handling done for value function")
        self.time_report.add_timer("recomputing reward")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        self.mega_gradient_diff = None
        self.mega_gradient_ = None
        self.mega_gradient = None
        def actor_closure():
            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            if self.log_gradients:
                actor_loss, actor_loss_, actor_loss_diff = self.compute_actor_loss()
            else:
                actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            if self.log_gradients:
                self.actor_optimizer.zero_grad()
                actor_loss_diff.backward()
                self.mega_gradient_diff = gather_all_gradients(self.actor)

                self.actor_optimizer.zero_grad()
                actor_loss_.backward()
                self.mega_gradient_ = gather_all_gradients(self.actor)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.mega_gradient = gather_all_gradients(self.actor)
            else:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(self.grad_norm_before_clip): #or self.grad_norm_before_clip > 1000000.:
                    print('NaN gradient')
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        if self.load_pretrain_dataset:
            self.min_replay = self.num_envs
        self.fill_replay_buffer()
        if self.generate_dataset_only:
            filename = 'dataset'
            torch.save([
                self.actor,
                self.dyn_rb.observations,
                self.dyn_rb.actions,
                self.dyn_rb.next_observations,
                self.dyn_rb.dones,
                self.dyn_rb.rewards,
                self.dyn_rb.p_ini_hidden_in,
                self.dyn_rb.num_envs,
                self.dyn_rb.pos,
                self.obs_rms,
                self.act_rms,
                self.ret_rms
            ], os.path.join(self.log_dir, "{}.pt".format(filename))) # /!\ Handle correctly self.prev_start_idx when reloading and restructuring this data
            self.close()
            return

        self.env.pretrained = True

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (self.actor_lr_schedule_min - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # dyn model training
            if not self.dyn_model_load:
                self.time_report.start_timer("dyn model training")
                self.train_dyn_model()
                self.time_report.end_timer("dyn model training")

            # train actor
            self.time_report.start_timer("actor training")
            actor_loss = self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            if self.train_value_function:
                with torch.no_grad():
                    self.compute_target_values()
                    dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last = False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            if self.train_value_function:
                for j in range(self.critic_iterations):
                    total_critic_loss = 0.
                    batch_cnt = 0
                    for i in range(len(dataset)):
                        batch_sample = dataset[i]
                        self.critic_optimizer.zero_grad()
                        training_critic_loss = self.compute_critic_loss(batch_sample)
                        training_critic_loss.backward()
                        
                        # ugly fix for simulation nan problem
                        for params in self.critic.parameters():
                            if params.grad is not None:
                                params.grad.nan_to_num_(0.0, 0.0, 0.0)

                        if self.truncate_grad:
                            clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                        self.critic_optimizer.step()

                        total_critic_loss += training_critic_loss
                        batch_cnt += 1
                    
                    self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                    print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')
            self.time_report.end_timer("critic training")

            self.iter_count += 1
            
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)

            if self.log_gradients:
                true_vs_demo_grad = F.cosine_similarity(self.mega_gradient_diff, self.mega_gradient, dim=0).cpu().item()
                true_vs_img_grad = F.cosine_similarity(self.mega_gradient_diff, self.mega_gradient_, dim=0).cpu().item()
                self.writer.add_scalar('gradients/true_vs_demo/step', true_vs_demo_grad, self.step_count)
                self.writer.add_scalar('gradients/true_vs_demo/iter', true_vs_demo_grad, self.iter_count)
                self.writer.add_scalar('gradients/true_vs_img/step', true_vs_img_grad, self.step_count)
                self.writer.add_scalar('gradients/true_vs_img/iter', true_vs_img_grad, self.iter_count)

            #self.writer.add_scalar('discarded_ratio/step', self.discarded_ratio, self.step_count) ##
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
                
                self.writer.add_scalar('policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                self.writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            if self.log_gradients:
                print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.10f}, grad norm after clip {:.2f}, dyn loss {:.2f}, true_vs_demo_gradnorm: {:.2f}, true_vs_img_gradnorm: {:.2f}'.format(\
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, self.grad_norm_before_clip, self.grad_norm_after_clip, self.dyn_loss, true_vs_demo_grad, true_vs_img_grad))
            else:
                print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.10f}, grad norm after clip {:.2f}, dyn loss {:.2f}'.format(\
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, self.grad_norm_before_clip, self.grad_norm_after_clip, self.dyn_loss))

            self.writer.flush()
        
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)

        self.time_report.end_timer("algorithm")

        self.time_report.report()
        
        self.save('final_policy')

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.log_dir, 'episode_loss_his.npy'), 'wb'), self.episode_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_discounted_loss_his.npy'), 'wb'), self.episode_discounted_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # evaluate the final policy's performance
        self.run(self.num_envs)

        self.close()
    
    def play(self, cfg):
        self.load(cfg['params']['general']['checkpoint'])
        self.run(cfg['params']['config']['player']['games_num'])
        
    def save(self, filename = None):
        if filename is None:
            filename = 'best_policy'
        torch.save([self.actor, self.critic, self.target_critic, self.dyn_model, self.obs_rms, self.act_rms, self.ret_rms], os.path.join(self.log_dir, "{}.pt".format(filename)))
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms = checkpoint[4].to(self.device)
        self.ret_rms = checkpoint[6].to(self.device) if checkpoint[6] is not None else checkpoint[6]
        
    def close(self):
        self.writer.close()
    
