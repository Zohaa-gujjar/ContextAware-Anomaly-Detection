from pathlib import Path  # Used to store filesystem locations safely.

from src.dataset.frame import Frame  # Represents one image frame in a sequence.
from src.dataset.label import Label  # Represents the annotations for a sequence.
from src.dataset.utils import list_image_files  # Finds and orders frame images.


class Sequence:
    # Groups a sequence's image frames and its matching label file.

    def __init__(self, sequence_id: str, frames_dir: Path, label_path: Path):
        # Store the sequence identifier used to distinguish this sample.
        self.sequence_id = sequence_id
        # Extract the scene ID from the sequence ID
        # Example: "13_007" -> 13
        self.scene_id = int(sequence_id.split("_")[0])
        # Store the directory containing the sequence's individual image frames.
        self.frames_dir = frames_dir
        # Store the path to the label/annotation file for this sequence.
        self.label_path = label_path

        # Build Frame objects immediately so the sequence is ready to use.
        self.frames = self.load_frames()
        # Load the associated labels once and keep them with the sequence.
        self.label = self.load_label()

    def load_frames(self):
        # Get supported image files in their expected numeric frame order.
        image_paths = list_image_files(self.frames_dir)

        # Wrap every image path in a Frame object for downstream processing.
        return [Frame(path) for path in image_paths]

    def load_label(self):
        # Create the label object from this sequence's annotation file.
        return Label(self.label_path)

    @property
    def total_frames(self):
        # Expose the frame count as a read-only computed attribute.
        return len(self.frames)

    def info(self):
        # Return a compact summary useful for logging or inspection.
        return {
    "scene_id": self.scene_id,
    "sequence_id": self.sequence_id,
    "total_frames": self.total_frames,
    "label_file": self.label.filename,
}

    def __repr__(self):
        # Provide a concise, readable representation for debugging sessions.
        return f"Sequence({self.sequence_id})"
