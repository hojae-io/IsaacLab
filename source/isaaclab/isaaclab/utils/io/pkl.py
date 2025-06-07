# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with pickle."""

import os
import pickle
from typing import Any, Union
from pathlib import Path


def load_pickle(filename: str) -> Any:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def dump_pickle(filename: Union[str, Path], data: Any) -> None:
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    # check ending
    path = Path(filename).with_suffix(".pkl")
    # create directory
    path.parent.mkdir(parents=True, exist_ok=True)
    # save data
    with path.open("wb") as f:
        pickle.dump(data, f)
