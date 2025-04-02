# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each reward term should instantiate the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # create buffer for managing reward per environment
        self._reward_buf: dict[str, torch.Tensor] = dict()
        # Buffer which stores the current step reward for each term for each environment
        self._step_reward: dict[str, torch.Tensor] = dict()
        # prepare extra info to store individual reward term information
        self._episode_sums: dict[dict[str: torch.Tensor]] = dict()
        for group_name in self._group_reward_term_names:
            self._reward_buf[group_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._step_reward[group_name] = torch.zeros((self.num_envs, len(self._group_reward_term_names[group_name])), dtype=torch.float, device=self.device)
            
            self._episode_sums[group_name] = dict()
            for term_name in self._group_reward_term_names[group_name]:
                self._episode_sums[group_name][term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self._group_reward_term_names)} groups.\n"

        # add info for each group
        for group_name in self._group_reward_term_names:
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Reward Terms in Group '{group_name}'"
            table.field_names = ["Index", "Name", "Weight"]
            # set alignment of table columns
            table.align["Name"] = "l"
            table.align["Weight"] = "r"
            # add info on each term
            for index, (name, term_cfg) in enumerate(zip(self._group_reward_term_names[group_name],
                                                         self._group_reward_term_cfgs[group_name])):
                table.add_row([index, name, term_cfg.weight])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active reward terms in each group.
        
        The keys are the group names and the values are the list of reward term names in the group.
        """
        return self._group_reward_term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for group_name in self._group_reward_term_names:
            for key in self._episode_sums[group_name].keys():
                # store information
                # r_1 + r_2 + ... + r_n
                episodic_sum_avg = torch.mean(self._episode_sums[group_name][key][env_ids])
                extras[f"Episode Reward/{group_name}/" + key] = episodic_sum_avg / self._env.max_episode_length_s
                # reset episodic sum
                self._episode_sums[group_name][key][env_ids] = 0.0
            # reset all the reward terms
            for term_cfg in self._group_reward_class_term_cfgs[group_name]:
                term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> dict[str, torch.Tensor]:
        """Compute the rewards per group for all groups.

        The method computes the rewards for all the groups handled by the reward manager.
        Please check the :meth:`compute_group' for computing rewards per group.

        Returns:
            A dictionary with keys as group names and values as the computed rewards.
        """
        # iterate over all the reward groups
        for group_name in self._group_reward_term_names:
            self.compute_group(group_name, dt)
        
        return self._reward_buf

    def compute_group(self, group_name: str, dt: float):
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            group_name: The name of the reward group.
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[group_name][:] = 0.0
        # iterate over all the reward terms
        for name, term_cfg in zip(self._group_reward_term_names[group_name], 
                                  self._group_reward_term_cfgs[group_name]):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf[group_name] += value
            # update episodic sum
            self._episode_sums[group_name][name] += value

            # Update current reward for this step.
            self._step_reward[group_name][:, self._group_reward_term_names[group_name].index(name)] = value / dt

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, group_name: str, term_name: str, cfg: RewardTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            group_name: The name of the reward group.
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_reward_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")
        if term_name not in self._group_reward_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._group_reward_term_cfgs[group_name][self._group_reward_term_names[group_name].index(term_name)] = cfg

    def get_term_cfg(self, group_name: str, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            group_name: The name of the reward group.
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_reward_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")
        if term_name not in self._group_reward_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._group_reward_term_cfgs[group_name][self._group_reward_term_names[group_name].index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for group_name in self._group_reward_term_names:
            for idx, name in enumerate(self._group_reward_term_names[group_name]):
                terms.append((name, [self._step_reward[group_name][env_idx, idx].cpu().item()]))
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to store information for each reward group
        self._group_reward_term_names: dict[str, list[str]] = dict()
        self._group_reward_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self._group_reward_class_term_cfgs: dict[str, list[RewardTermCfg]] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # initialize list for the group settings
            self._group_reward_term_names[group_name] = list()
            self._group_reward_term_cfgs[group_name] = list()
            self._group_reward_class_term_cfgs[group_name] = list()

            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items() 

            # iterate over all the terms
            for term_name, term_cfg in group_cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # check for valid weight type
                if not isinstance(term_cfg.weight, (float, int)):
                    raise TypeError(
                        f"Weight for the term '{term_name}' is not of type float or int."
                        f" Received: '{type(term_cfg.weight)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add function to list
                self._group_reward_term_names[group_name].append(term_name)
                self._group_reward_term_cfgs[group_name].append(term_cfg)
                # check if the term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_reward_class_term_cfgs[group_name].append(term_cfg)
