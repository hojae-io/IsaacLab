"""
Base recorder for logging images and creating videos.
"""

import os
import cv2
import numpy as np

from abc import ABC, abstractmethod

class RecorderBase(ABC):
    def __init__(self, fps: int):
        self.fps = fps
        self.checkpoint = None
        self.folderpath = None

    @abstractmethod
    def __str__(self) -> str:
        """Subclasses must implement string representation."""
        pass

    @abstractmethod
    def log(self, *args, **kwargs):
        """Subclasses must implement logging behavior."""
        pass

    @abstractmethod
    def save(self, resume_path: str):
        """Subclasses must implement save behavior (e.g., saving video, logs, plots)."""
        pass

class VideoRecorder(RecorderBase):

    def __init__(self, fps: int = 50):
        super().__init__(fps)
        self.frames = []

    def __str__(self) -> str:
        """Return the string representation of the recorder."""
        msg = "VideoRecorder"
        msg += f" (fps: {self.fps})"
        return msg

    def log(self, image: np.ndarray, *args):
        # * Log images
        self.frames.append(image)

    def save(self, resume_path: str):
        self.setup_save_folder(resume_path)
        self.save_video()

    def setup_save_folder(self, resume_path: str):
        """Set up the checkpoint and the folder path for saving the video."""
        path_split = resume_path.split('/')
        self.checkpoint = path_split[-1][:-3]
        log_root_path = '/'.join(path_split[:-1])
        self.folderpath = os.path.join(log_root_path, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)

    def save_video(self):
        print("Creating video...")
        filepath = os.path.join(self.folderpath, f"{self.checkpoint}.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frames[0].shape[1], self.frames[0].shape[0]))

        # Write the frames to the video file
        for frame in self.frames:
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        # Release the video writer and print completion message
        out.release()
        print(f"Video saved to {filepath}")