# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import convert_quat

from ..sensor_base import SensorBase
from .contact_sensor_data import ContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class ContactSensor(SensorBase):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body in the world frame.
    It relies on the `PhysX ContactReporter`_ API to be activated on the rigid bodies.

    To enable the contact reporter on a rigid body, please make sure to enable the
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors` on your
    asset spawner configuration. This will enable the contact reporter on all the rigid bodies
    in the asset.

    The sensor can be configured to report the contact forces on a set of bodies with a given
    filter pattern using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is useful
    when you want to report the contact forces between the sensor bodies and a specific set of
    bodies in the scene. The data can be accessed using the :attr:`ContactSensorData.force_matrix_w`.
    Please check the documentation on `RigidContact`_ for more details.

    The reporting of the filtered contact forces is only possible as one-to-many. This means that only one
    sensor body in an environment can be filtered against multiple bodies in that environment. If you need to
    filter multiple sensor bodies against multiple bodies, you need to create separate sensors for each sensor
    body.

    As an example, suppose you want to report the contact forces for all the feet of a robot against an object
    exclusively. In that case, setting the :attr:`ContactSensorCfg.prim_path` and
    :attr:`ContactSensorCfg.filter_prim_paths_expr` with ``{ENV_REGEX_NS}/Robot/.*_FOOT`` and ``{ENV_REGEX_NS}/Object``
    respectively will not work. Instead, you need to create a separate sensor for each foot and filter
    it against the object.

    .. _PhysX ContactReporter: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_contact_report_a_p_i.html
    .. _RigidContact: https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.core/docs/index.html#isaacsim.core.prims.RigidContact
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # initialize self._body_physx_view for running in extension mode
        self._body_physx_view = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self.body_physx_view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self.num_bodies}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self.body_physx_view.count

    @property
    def data(self) -> ContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the rigid bodies captured (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._body_physx_view

    @property
    def contact_physx_view(self) -> physx.RigidContactView:
        """Contact reporter view for the bodies (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._contact_physx_view

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids] = 0.0
        self._data.net_forces_w_history[env_ids] = 0.0
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = 0.0
        # reset force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            self._data.force_matrix_w[env_ids] = 0.0
            if self.cfg.max_contact_data_count_per_env > 0:
                self._data.contact_forces_buffer[env_ids] = 0.0
                self._data.contact_points_buffer[env_ids] = 0.0
                self._data.contact_normals_buffer[env_ids] = 0.0
                self._data.contact_separation_distances_buffer[env_ids] = 0.0
                self._data.contact_count_buffer[env_ids] = 0
                self._data.contact_start_indices_buffer[env_ids] = 0

                self._data.friction_forces_buffer[env_ids] = 0.0
                self._data.friction_points_buffer[env_ids] = 0.0
                self._data.friction_count_buffer[env_ids] = 0
                self._data.friction_start_indices_buffer[env_ids] = 0

                self._data.GRF_forces_buffer[env_ids] = 0.0
                self._data.GRF_points_buffer[env_ids] = 0.0
                self._data.GRF_count_buffer[env_ids] = 0
        # reset the current air time
        if self.cfg.track_air_time:
            self._data.current_air_time[env_ids] = 0.0
            self._data.last_air_time[env_ids] = 0.0
            self._data.current_contact_time[env_ids] = 0.0
            self._data.last_contact_time[env_ids] = 0.0

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact.

        Note:
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have established contact within the last
            :attr:`dt` seconds. Shape is (N, B), where N is the number of sensors and B is the
            number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the bodies are in contact
        currently_in_contact = self.data.current_contact_time > 0.0
        less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have broken contact within the last :attr:`dt` seconds.

        This function checks if the bodies have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the bodies are considered to not be in contact.

        Note:
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the sensor is configured to track contact time
        currently_detached = self.data.current_air_time > 0.0
        less_than_dt_detached = self.data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check that only rigid bodies are selected
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = list()
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            # check if prim has contact reporter API
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                prim_path = prim.GetPath().pathString
                body_names.append(prim_path.rsplit("/", 1)[-1])
        # check that there is at least one body with contact reporter API
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        # construct regex expression for the body names
        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        # convert regex expressions to glob expressions for PhysX
        body_names_glob = body_names_regex.replace(".*", "*")
        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]

        # create a rigid prim view for the sensor
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
        self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
            body_names_glob, filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_env * self._num_envs,
        )
        # resolve the true count of bodies
        self._num_bodies = self.body_physx_view.count // self._num_envs
        # check that contact reporter succeeded
        if self._num_bodies != len(body_names):
            raise RuntimeError(
                "Failed to initialize contact reporter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

        # prepare data buffers
        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        # optional buffers
        # -- history of net forces
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history = torch.zeros(
                self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
            )
        else:
            self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
        # -- pose of sensor origins
        if self.cfg.track_pose:
            self._data.pos_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
            self._data.quat_w = torch.zeros(self._num_envs, self._num_bodies, 4, device=self._device)
        # -- air/contact time between contacts
        if self.cfg.track_air_time:
            self._data.last_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.last_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
        # force matrix: (num_envs, num_bodies, num_filter_shapes, 3)
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_filters = self.contact_physx_view.filter_count
            self._data.force_matrix_w = torch.zeros(
                self._num_envs, self._num_bodies, num_filters, 3, device=self._device
            )

            if self.cfg.max_contact_data_count_per_env > 0:
                # * Assume 1 sensor (foot) / 1 filter (ground) per environment
                self._data.contact_forces_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 1, device=self._device)
                self._data.contact_points_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.contact_normals_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.contact_separation_distances_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 1, device=self._device)
                self._data.contact_count_buffer = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
                self._data.contact_start_indices_buffer = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)

                self._data.friction_forces_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.friction_points_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.friction_count_buffer = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
                self._data.friction_start_indices_buffer = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)

                self._data.GRF_forces_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.GRF_points_buffer = torch.zeros(self._num_envs, self.cfg.max_contact_data_count_per_env, 3, device=self._device)
                self._data.GRF_count_buffer = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # obtain the contact forces
        # TODO: We are handling the indexing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self.contact_physx_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]
        # update contact force history
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids, 1:] = self._data.net_forces_w_history[env_ids, :-1].clone()
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self.contact_physx_view.filter_count
            # acquire and shape the force matrix
            force_matrix_w = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
            if self.cfg.max_contact_data_count_per_env > 0:
                (   contact_forces, 
                    contact_points, 
                    contact_normals, 
                    contact_separation_distances, 
                    contact_count,
                    contact_start_indices, 
                ) = self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)

                rel_idx = torch.arange(self.cfg.max_contact_data_count_per_env, device=self.device).unsqueeze(0)  # Shape: (1, max_contacts)
                contact_mask = rel_idx < contact_count[env_ids] # Shape: (len(env_ids), max_contacts)
                contact_full_idx = contact_start_indices[env_ids] + rel_idx  # Shape: (len(env_ids), max_contacts)
                
                contact_forces_buffer = self._data.contact_forces_buffer[env_ids]
                contact_points_buffer = self._data.contact_points_buffer[env_ids]
                contact_normals_buffer = self._data.contact_normals_buffer[env_ids]
                contact_separation_distances_buffer = self._data.contact_separation_distances_buffer[env_ids]
                contact_count_buffer = self._data.contact_count_buffer[env_ids]
                contact_start_indices_buffer = self._data.contact_start_indices_buffer[env_ids]

                contact_forces_buffer[contact_mask] = contact_forces[contact_full_idx[contact_mask]]
                contact_forces_buffer[~contact_mask] = 0.0
                contact_points_buffer[contact_mask] = contact_points[contact_full_idx[contact_mask]]
                contact_points_buffer[~contact_mask] = 0.0
                contact_normals_buffer[contact_mask] = contact_normals[contact_full_idx[contact_mask]]
                contact_normals_buffer[~contact_mask] = 0.0
                contact_separation_distances_buffer[contact_mask] = contact_separation_distances[contact_full_idx[contact_mask]]
                contact_separation_distances_buffer[~contact_mask] = 0.0
                contact_count_buffer = contact_count[env_ids].squeeze(1)
                contact_start_indices_buffer = contact_start_indices[env_ids].squeeze(1)

                self._data.contact_forces_buffer[env_ids] = contact_forces_buffer
                self._data.contact_points_buffer[env_ids] = contact_points_buffer
                self._data.contact_normals_buffer[env_ids] = contact_normals_buffer
                self._data.contact_separation_distances_buffer[env_ids] = contact_separation_distances_buffer
                self._data.contact_count_buffer[env_ids] = contact_count_buffer
                self._data.contact_start_indices_buffer[env_ids] = contact_start_indices_buffer

                (   friction_forces,
                    friction_points,
                    friction_count,
                    friction_start_indices,
                ) = self.contact_physx_view.get_friction_data(dt=self._sim_physics_dt)
                
                friction_mask = rel_idx < friction_count[env_ids] # Shape: (len(env_ids), max_contacts)
                friction_full_idx = friction_start_indices[env_ids] + rel_idx  # Shape: (len(env_ids), max_contacts)

                friction_forces_buffer = self._data.friction_forces_buffer[env_ids]
                friction_points_buffer = self._data.friction_points_buffer[env_ids]
                friction_count_buffer = self._data.friction_count_buffer[env_ids]
                friction_start_indices_buffer = self._data.friction_start_indices_buffer[env_ids]

                friction_forces_buffer[friction_mask] = friction_forces[friction_full_idx[friction_mask]]
                friction_forces_buffer[~friction_mask] = 0.0
                friction_points_buffer[friction_mask] = friction_points[friction_full_idx[friction_mask]]
                friction_points_buffer[~friction_mask] = 0.0
                friction_count_buffer = friction_count[env_ids].squeeze(1)
                friction_start_indices_buffer = friction_start_indices[env_ids].squeeze(1)

                self._data.friction_forces_buffer[env_ids] = friction_forces_buffer
                self._data.friction_points_buffer[env_ids] = friction_points_buffer
                self._data.friction_count_buffer[env_ids] = friction_count_buffer
                self._data.friction_start_indices_buffer[env_ids] = friction_start_indices_buffer

                # Post-processing for Ground Reaction Forces (GRF) = Concatenation of friction (f_x, f_y) and contact (f_z) forces

                GRF_forces_buffer = self._data.GRF_forces_buffer[env_ids]
                GRF_points_buffer = self._data.GRF_points_buffer[env_ids]
                GRF_count_buffer = self._data.GRF_count_buffer[env_ids]

                tol = 1e-2
                diff = friction_points_buffer.unsqueeze(2) - contact_points_buffer.unsqueeze(1)
                distances = diff.norm(dim=-1)
                mask = distances < tol
                contact_force_z_sum = (contact_forces_buffer.unsqueeze(1) * mask.unsqueeze(-1)).sum(dim=2)  # shape: (N, F, 1)

                GRF_forces_buffer = friction_forces_buffer.clone()
                GRF_forces_buffer[..., 2:3] = contact_force_z_sum

                GRF_points_buffer = friction_points_buffer.clone()
                GRF_count_buffer = friction_count_buffer.clone()

                self._data.GRF_forces_buffer[env_ids] = GRF_forces_buffer
                self._data.GRF_points_buffer[env_ids] = GRF_points_buffer
                self._data.GRF_count_buffer[env_ids] = GRF_count_buffer

        # obtain the pose of the sensor origin
        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
            pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

        # obtain the air time
        if self.cfg.track_air_time:
            # -- time elapsed since last update
            # since this function is called every frame, we can use the difference to get the elapsed time
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            # -- check contact state of bodies
            is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
            # -- update the last contact time if body has just become in contact
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            # -- increment time for bodies that are not in contact
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )
            # -- update the last contact time if body has just detached
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            # -- increment time for bodies that are in contact
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self.body_physx_view is None:
            return
        if self.cfg.max_contact_data_count_per_env == 0:
            # marker indices
            # 0: contact, 1: no contact
            net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
            marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
            # check if prim is visualized
            if self.cfg.track_pose:
                frame_origins: torch.Tensor = self._data.pos_w
            else:
                pose = self.body_physx_view.get_transforms()
                frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
            # visualize
            self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))
        else:   
            """ Detailed contact data is available
                Origin of the arrow is the contact points
                Direction of the arrow is the (friction_forces_x, friction_forces_y, contact forces)
            """
            rel_idx = torch.arange(self.cfg.max_contact_data_count_per_env, device=self.device).unsqueeze(0)  # Shape: (1, max_contacts)
            mask = rel_idx < self._data.GRF_count_buffer.unsqueeze(1)  # Shape: (N, K)

            GRF_norm_vector = torch.nn.functional.normalize(self._data.GRF_forces_buffer[mask])
            base_vector = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(GRF_norm_vector.shape[0], 1)
            axis = torch.cross(base_vector, GRF_norm_vector, dim=1)
            angle = torch.acos(torch.clamp(torch.sum(base_vector * GRF_norm_vector, dim=1), -1.0, 1.0))
            GRF_arrow_quat = math_utils.quat_from_angle_axis(angle, axis)

            default_scale = self.contact_visualizer.cfg.markers["arrow"].scale
            GRF_arrow_scale = torch.tensor(default_scale, device=self.device).repeat(GRF_norm_vector.shape[0], 1)
            GRF_arrow_scale[:, 0] = (self._data.GRF_forces_buffer[mask].norm(dim=1) * 0.05)

            local_offset = 0.1 * GRF_arrow_scale[:, 0:1] * torch.tensor([0.25, 0.0, 0.0], device=self.device) # The given pos divides the arrow 3:1 = head:tail
            transformed_offset = (math_utils.matrix_from_quat(GRF_arrow_quat) @ local_offset.unsqueeze(-1)).squeeze(-1)
            GRF_arrow_pos_w = self._data.friction_points_buffer[mask] + transformed_offset

            self.contact_visualizer.visualize(GRF_arrow_pos_w, GRF_arrow_quat, GRF_arrow_scale)


    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._body_physx_view = None
        self._contact_physx_view = None
