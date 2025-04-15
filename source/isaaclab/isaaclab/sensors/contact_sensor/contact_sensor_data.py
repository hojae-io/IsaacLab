# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ContactSensorData:
    """Data container for the contact reporting sensor."""

    pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """

    quat_w: torch.Tensor | None = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """

    net_forces_w: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, B, 3), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    net_forces_w_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, B, 3), where N is the number of sensors, T is the configured history length
    and B is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    force_matrix_w: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and ``M`` is the number of filtered bodies.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """

    last_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air before the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    current_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air since the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """


    # * Data container for contact_physx_view.get_contact_data(dt)

    contact_forces_buffer: torch.Tensor | None = None
    """Buffer storing detailed contact forces. 

    Shape is (num_envs, max_contact_data_count_per_env, 1), where each entry is the force magnitude at a contact point.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    contact_points_buffer: torch.Tensor | None = None
    """Buffer storing contact points in world frame. 

    Shape is (num_envs, max_contact_data_count_per_env, 3), where each entry is the (x, y, z) position of a contact point.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    contact_normals_buffer: torch.Tensor | None = None
    """Buffer storing contact normals in world frame. 

    Shape is (num_envs, max_contact_data_count_per_env, 3), where each entry is the (x, y, z) normal vector at a contact point.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    contact_separation_distances_buffer: torch.Tensor | None = None
    """Buffer storing separation distances at contact points.

    Shape is (num_envs, max_contact_data_count_per_env, 1), where each entry is the separation distance.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    contact_count_buffer: torch.Tensor | None = None
    """Number of active contacts per sensor-filter pair. 

    Shape is (num_envs, sensor_count_per_env, filter_count), where each entry is the count of contacts.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
        """

    contact_start_indices_buffer: torch.Tensor | None = None
    """Start indices in buffers for each sensor-filter pair.

    Shape is (num_envs, sensor_count_per_env, filter_count), where each entry points to the start index in data buffers. 

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
        """


    # * Data container for contact_physx_view.get_friction_data(dt)

    friction_forces_buffer: torch.Tensor | None = None
    """Buffer storing friction forces at contact points.
    
    Shape is (num_envs, max_contact_data_count_per_env, 3), where each entry is the (x, y, z) friction force vector at a contact point.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    friction_points_buffer: torch.Tensor | None = None
    """Buffer storing friction points in world frame.

    Shape is (num_envs, max_contact_data_count_per_env, 3), where each entry is the (x, y, z) position of a friction point.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    friction_count_buffer: torch.Tensor | None = None
    """Number of active friction contacts per sensor-filter pair.

    Shape is (num_envs, sensor_count_per_env, filter_count), where each entry is the count of friction contacts.
    
    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """

    friction_start_indices_buffer: torch.Tensor | None = None
    """Start indices in buffers for each sensor-filter pair.

    Shape is (num_envs, sensor_count_per_env, filter_count), where each entry points to the start index in data buffers.
    
    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        If the :attr: `ContactSensorCfg.max_contact_data_count` is 0, then this quantity is None.
    """