# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recording wrapper for Isaac Lab vector environments.

Unlike :class:`gymnasium.wrappers.RecordVideo`, this wrapper does **not** assume a
Gymnasium ``step`` return of ``(obs, rew, terminated, truncated, info)``. It forwards
:class:`~isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv`-style **six**-tuple
returns unchanged, while capturing frames via :meth:`gymnasium.Env.render` when
``render_mode`` supports it (typically ``rgb_array``).

Dependencies: ``moviepy`` (same as Gymnasium's ``RecordVideo``), ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np


class RecordVideoWrapper(gym.Wrapper):
    """Record viewport (or sensor) RGB frames to MP4 during training.

    API mirrors :class:`gymnasium.wrappers.RecordVideo` kwargs where possible so
    training scripts can pass the same ``video_kwargs`` dict.

    Args:
        env: Isaac Lab environment (already created with ``render_mode="rgb_array"`` when recording).
        video_folder: Directory to write ``*.mp4`` files.
        step_trigger: If set, ``step_trigger(step_id)`` starts a new clip when True.
        episode_trigger: If set, ``episode_trigger(episode_id)`` starts a clip after each reset when True.
        video_length: Max frames per clip; ``0`` means unlimited until stopped.
        name_prefix: Filename prefix for saved videos.
        disable_logger: If True, pass ``logger=None`` to moviepy for quieter output.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        super().__init__(env)
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = os.path.abspath(video_folder)
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_length = video_length if video_length != 0 else float("inf")
        self.name_prefix = name_prefix
        self.disable_logger = disable_logger

        fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = int(fps) if fps is not None else 30

        self._recording = False
        self._video_name: str | None = None
        self.recorded_frames: list[np.ndarray] = []
        self.step_id = -1
        self.episode_id = -1

    @property
    def recording(self) -> bool:
        return self._recording

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_id += 1

        if self._recording and self.video_length == float("inf"):
            self.stop_recording()

        obs, info = self.env.reset(seed=seed, options=options)

        if self.episode_trigger is not None and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        if self._recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, info

    def step(self, action):
        # Match gymnasium.wrappers.RecordVideo: increment step counter after env.step.
        step_out = self.env.step(action)
        self.step_id += 1

        if self.step_trigger is not None and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if self._recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return step_out

    def render(self):
        return self.env.render()

    def _capture_frame(self) -> None:
        if not self._recording:
            return
        frame = self.env.render()
        if isinstance(frame, list):
            if len(frame) == 0:
                return
            frame = frame[-1]
        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()

    def start_recording(self, video_name: str) -> None:
        if self._recording:
            self.stop_recording()
        self._recording = True
        self._video_name = video_name
        self.recorded_frames = []

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        frames = self.recorded_frames
        self.recorded_frames = []
        name = self._video_name or f"{self.name_prefix}-clip"
        self._video_name = None

        if not frames:
            return

        try:
            from moviepy import ImageSequenceClip
        except ImportError as e:
            raise ImportError(
                'Recording requires moviepy. Install with: pip install "gymnasium[other]" or pip install moviepy'
            ) from e

        # moviepy expects uint8 RGB, (H, W, 3)
        clip = ImageSequenceClip(frames, fps=self.frames_per_sec)
        out_path = os.path.join(self.video_folder, f"{name}.mp4")
        try:
            # MoviePy 2.x / Gymnasium style: logger=...; MoviePy 1.x: verbose=...
            try:
                clip.write_videofile(out_path, logger=None if self.disable_logger else "bar")
            except TypeError:
                clip.write_videofile(out_path, verbose=not self.disable_logger)
        finally:
            clip.close()

    def close(self):
        if self._recording:
            self.stop_recording()
        super().close()
