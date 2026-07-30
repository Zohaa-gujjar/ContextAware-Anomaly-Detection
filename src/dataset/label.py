from pathlib import Path
import numpy as np


class Label:
    """
    Represents the ground-truth label of a sequence.
    """

    def __init__(self, label_path: Path):
        self.label_path = label_path

    @property
    def filename(self):
        return self.label_path.name

    @property
    def sequence_id(self):
        return self.label_path.stem

    def load(self):
        """
        Load the label array.
        """
        return np.load(self.label_path)

    def __repr__(self):
        return f"Label({self.filename})"