# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab–compatible training video recording (no Gymnasium 5-tuple ``step`` API)."""

from .record_video_wrapper import RecordVideoWrapper

__all__ = ["RecordVideoWrapper"]
