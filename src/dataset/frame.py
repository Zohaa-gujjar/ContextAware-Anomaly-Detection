from pathlib import Path
import cv2


class Frame:
    """
    Represents a single image frame.
    """

    def __init__(self, image_path: Path):
        self.image_path = image_path

    @property
    def filename(self):
        return self.image_path.name

    @property
    def frame_id(self):
        return self.image_path.stem

    def load(self):
        """
        Load the image using OpenCV.
        """
        return cv2.imread(str(self.image_path))

    def __repr__(self):
        return f"Frame({self.filename})"