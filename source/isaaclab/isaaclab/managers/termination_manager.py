# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination manager for computing done signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerminationManager(ManagerBase):
    """Manager for computing done signals for a given world.

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which takes the environment as an
    argument and returns a boolean tensor of shape (num_envs,). The termination manager
    computes the termination signal as the union (logical or) of all the termination terms.

    Following the `Gymnasium API <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_,
    the termination signal is computed as the logical OR of the following signals:

    * **Time-out**: This signal is set to true if the environment has ended after an externally defined condition
      (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
      timed out (i.e. reached max episode length).
    * **Terminated**: This signal is set to true if the environment has reached a terminal state defined by the
      environment. This state may correspond to task success, task failure, robot falling, etc.

    These signals can be individually accessed using the :attr:`time_outs` and :attr:`terminated` properties.

    The termination terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each termination term should instantiate the :class:`TerminationTermCfg` class. The term's
    configuration :attr:`TerminationTermCfg.time_out` decides whether the term is a timeout or a termination term.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initializes the termination manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, TerminationTermCfg]``).
            env: An environment object.
        """
        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # create buffer for managing termination per environment
        self._reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._time_outs_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros_like(self._time_outs_buf)

        # prepare extra info to store individual termination term information
        self._group_termination_dones = dict()
        self._group_termination_term_dones = dict()
        for group_name in self._group_termination_term_names:
            self._group_termination_dones[group_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._group_termination_term_dones[group_name] = dict()
            for term_name in self._group_termination_term_names[group_name]:
                self._group_termination_term_dones[group_name][term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def __str__(self) -> str:
        """Returns: A string representation for termination manager."""
        msg = f"<TerminationManager> contains {len(self._group_termination_term_names)} groups.\n"

        # add info for each group
        for group_name in self._group_termination_term_names:
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Termination Terms in Group: '{group_name}'"
            table.field_names = ["Index", "Name", "Time Out"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info on each term
            for index, (name, term_cfg) in enumerate(zip(self._group_termination_term_names[group_name], 
                                                         self._group_termination_term_cfgs[group_name])):
                table.add_row([index, name, term_cfg.time_out])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active termination terms in each group.
        
        The keys are the group names and the values are the list of termination term names in the group.
        """
        return self._group_termination_term_names

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is (num_envs,)."""
        return self._reset_buf

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,).

        This signal is set to true if the environment has ended after an externally defined condition
        (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
        timed out (i.e. reached max episode length).
        """
        return self._group_termination_dones['time_out']

    @property
    def terminated(self) -> dict[str, torch.Tensor]:
        """The terminated signal (reaching a terminal state). Shape is {term_name: (num_envs,)}.

        This signal is set to true if the environment has reached a terminal state defined by the environment.
        This state may correspond to task success, task failure, robot falling, etc.
        """
        return self._group_termination_dones

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual termination terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        for group_name in self._group_termination_term_names:
            for term_name in self._group_termination_term_names[group_name]:
                extras[f"Episode Termination/{group_name}/{term_name}"] = torch.count_nonzero(self._group_termination_term_dones[group_name][term_name][env_ids]).item()
            # reset all the termination terms
            for term_cfg in self._group_termination_class_term_cfgs[group_name]:
                term_cfg.func.reset(env_ids=env_ids)

        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """Compute the terminations per group for all groups.

        The method computes the terminations for all the groups handled by the termination manager.
        Please check the :meth:`compute_group` on the processing of terminations per group.

        Returns:
            A dictionary with keys as the group names and values as the computed terminations.
        """
        # create a buffer for storing terminations from all the groups
        self._reset_buf[:] = False
        # iterate over all the terms in each group
        for group_name in self._group_termination_term_names:
            self._group_termination_dones[group_name] = self.compute_group(group_name)
            self._reset_buf |= self._group_termination_dones[group_name]
        # return the computed terminations
        return self._reset_buf

    def compute_group(self, group_name: str) -> torch.Tensor:
        """Compute the terminations for the specified group.

        This function calls each termination term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        # reset computation
        self._time_outs_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the termination terms
        for term_name, term_cfg in zip(self._group_termination_term_names[group_name], 
                                       self._group_termination_term_cfgs[group_name]):
            value = term_cfg.func(self._env, **term_cfg.params)
            # store timeout signal separately
            if term_cfg.time_out:
                self._time_outs_buf |= value
            else:
                self._terminated_buf |= value
            # add to episode dones
            self._group_termination_term_dones[group_name][term_name][:] = value
        # return combined termination signal
        return self._time_outs_buf | self._terminated_buf

    def get_term(self, group_name: str, term_name: str) -> torch.Tensor:
        """Returns the termination term with the specified name.

        Args:
            group_name: The name of the termination group.
            term_name: The name of the termination term.

        Returns:
            The corresponding termination term value. Shape is (num_envs,).
        """
        return self._group_termination_term_dones[group_name][term_name]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for group_name in self._group_termination_term_names:
            for term_name in self._group_termination_term_names[group_name]:
                terms.append((term_name, [self._group_termination_term_dones[group_name][term_name][env_idx].float().cpu().item()]))

        return terms

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, group_name: str, term_name: str, cfg: TerminationTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            group_name: The name of the termination group.
            term_name: The name of the termination term.
            cfg: The configuration for the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_termination_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self._group_termination_term_names[group_name]:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # set the configuration
        self._group_termination_term_cfgs[group_name][self._group_termination_term_names[group_name].index(term_name)] = cfg

    def get_term_cfg(self, group_name: str, term_name: str) -> TerminationTermCfg:
        """Gets the configuration for the specified term.

        Args:
            group_name: The name of the termination group.
            term_name: The name of the termination term.

        Returns:
            The configuration of the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_termination_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self._group_termination_term_names[group_name]:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # return the configuration
        return self._group_termination_term_cfgs[group_name][self._group_termination_term_names[group_name].index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of termination terms functions."""
        # create buffers to store information for each termination group
        self._group_termination_term_names: dict[str, list[str]] = dict()
        self._group_termination_term_cfgs: dict[str, list[TerminationTermCfg]] = dict()
        self._group_termination_class_term_cfgs: dict[str, list[TerminationTermCfg]] = dict()

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
            self._group_termination_term_names[group_name] = list()
            self._group_termination_term_cfgs[group_name] = list()
            self._group_termination_class_term_cfgs[group_name] = list()

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
                if not isinstance(term_cfg, TerminationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type TerminationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add function to list
                self._group_termination_term_names[group_name].append(term_name)
                self._group_termination_term_cfgs[group_name].append(term_cfg)
                # check if the term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_termination_class_term_cfgs[group_name].append(term_cfg)
