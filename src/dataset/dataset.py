from pathlib import Path

from src.dataset.sequence import Sequence


class Dataset:
    """
    Represents the complete ShanghaiTech dataset.

    A Dataset object contains multiple Sequence objects.
    Each Sequence represents one surveillance video.
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize the Dataset.

        Parameters
        ----------
        dataset_path : Path
            Path to the SHANGHAI_TRAIN directory.
        """

        self.dataset_path = dataset_path

        # Paths to dataset subdirectories
        self.frames_path = dataset_path / "frames"
        self.labels_path = dataset_path / "label"

        # Load all sequences in the dataset
        self.sequences = self.load_sequences()

    @classmethod
    def from_directory(cls, dataset_path: Path):
        """
        Factory method for creating a Dataset object.

        Example
        -------
        dataset = Dataset.from_directory(path)
        """

        return cls(dataset_path)

    def load_sequences(self):
        """
        Scan the dataset folders and create Sequence objects.

        Every folder inside 'frames' corresponds to one video sequence.
        The matching label is searched inside the 'label' directory.
        """

        sequences = []

        # Iterate through every sequence folder
        for sequence_folder in sorted(self.frames_path.iterdir()):

            # Ignore files (only process folders)
            if not sequence_folder.is_dir():
                continue

            # Example: 13_007
            sequence_id = sequence_folder.name

            # Matching label file
            label_file = self.labels_path / f"{sequence_id}.npy"

            # Create a Sequence object
            sequence = Sequence(
                sequence_id=sequence_id,
                frames_dir=sequence_folder,
                label_path=label_file,
            )

            sequences.append(sequence)

        return sequences

    def get_sequence(self, sequence_id: str):
        """
        Return the Sequence object corresponding to the given sequence ID.

        Parameters
        ----------
        sequence_id : str
            Example: "13_007"

        Returns
        -------
        Sequence
            Matching Sequence object.

        Raises
        ------
        ValueError
            If the sequence does not exist.
        """

        for sequence in self.sequences:
            if sequence.sequence_id == sequence_id:
                return sequence

        raise ValueError(f"Sequence '{sequence_id}' not found.")

    @property
    def total_sequences(self):
        """
        Return the total number of sequences.
        """

        return len(self.sequences)

    @property
    def total_scenes(self):
        """
        Return the total number of unique scenes in the dataset.
        """

        # Create a set of unique scene IDs
        return len({sequence.scene_id for sequence in self.sequences})

    def info(self):
        """
        Return basic information about the dataset.
        """

        return {
            "dataset_path": str(self.dataset_path),
            "total_sequences": self.total_sequences,
            "total_scenes": self.total_scenes,
        }

    def __repr__(self):
        """
        String representation of the Dataset object.
        """

        return f"Dataset({self.total_sequences} sequences)"