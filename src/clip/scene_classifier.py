import torch
from pathlib import Path

from src.clip.clip_model import CLIPModelWrapper


class SceneClassifier:
    """
    Classifies ShanghaiTech frames into one of the 13 scenes
    using CLIP image embeddings.

    The classifier creates one representative embedding
    (centroid) for each scene and compares new frames
    against those scene embeddings.
    """

    def __init__(self, clip_model: CLIPModelWrapper):
        """
        Initialize the SceneClassifier.

        Parameters
        ----------
        clip_model : CLIPModelWrapper
            Loaded CLIP model used to generate image embeddings.
        """

        self.clip_model = clip_model

        # Stores one representative embedding for each scene.
        # Example:
        # {
        #     1: tensor(...),
        #     2: tensor(...),
        #     ...
        # }
        self.scene_embeddings = {}

    def build_scene_embeddings(
        self,
        dataset_path: Path,
        samples_per_sequence: int = 5,
    ):
        """
        Build representative CLIP embeddings for all 13 scenes.

        Frames are sampled from the reference sequences belonging
        to each scene.

        Parameters
        ----------
        dataset_path : Path
            Path to the SHANGHAI_TRAIN directory.

        samples_per_sequence : int
            Number of frames sampled from each sequence.
        """

        frames_path = dataset_path / "frames"

        # Find all sequence folders.
        sequence_folders = sorted(
            folder
            for folder in frames_path.iterdir()
            if folder.is_dir()
        )

        # Group sequence folders according to their scene ID.
        scenes = {}

        for sequence_folder in sequence_folders:

            sequence_id = sequence_folder.name

            # ShanghaiTech sequence IDs follow the format:
            # 01_002, 02_014, 13_007, etc.
            scene_id = int(sequence_id.split("_")[0])

            scenes.setdefault(scene_id, []).append(sequence_folder)

        # Process every scene.
        for scene_id, sequences in sorted(scenes.items()):

            print(f"Building embedding for Scene {scene_id}...")

            embeddings = []

            for sequence_folder in sequences:

                # Get all JPG frames from this sequence.
                frame_files = sorted(
                    sequence_folder.glob("*.jpg")
                )

                if not frame_files:
                    continue

                # Select evenly spaced frames instead of processing
                # every frame in the sequence.
                sample_count = min(
                    samples_per_sequence,
                    len(frame_files)
                )

                indices = torch.linspace(
                    0,
                    len(frame_files) - 1,
                    steps=sample_count
                ).long()

                for index in indices:

                    frame_path = frame_files[index.item()]

                    # Load the image through PIL.
                    image = self._load_image(frame_path)

                    # Skip corrupted or unreadable images.
                    if image is None:
                     continue

                    # Convert image into a CLIP embedding.
                    embedding = self.clip_model.encode_image(image)

                    embeddings.append(embedding)

            if not embeddings:
                continue

            # Combine all sampled embeddings.
            embeddings = torch.cat(embeddings, dim=0)

            # Calculate the average embedding for this scene.
            scene_embedding = embeddings.mean(dim=0, keepdim=True)

            # Normalize the scene representation.
            scene_embedding = scene_embedding / scene_embedding.norm(
                dim=-1,
                keepdim=True
            )

            self.scene_embeddings[scene_id] = scene_embedding

            print(
                f"Scene {scene_id}: "
                f"{len(embeddings)} frame embeddings collected."
            )

    def predict(self, image):
        """
        Predict the scene ID for a given image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input ShanghaiTech frame.

        Returns
        -------
        int
            Predicted scene ID.
        """

        # Extract CLIP embedding for the input image.
        image_embedding = self.clip_model.encode_image(image)

        best_scene = None
        best_similarity = -1.0

        # Compare the input embedding with every scene embedding.
        for scene_id, scene_embedding in self.scene_embeddings.items():

            # Cosine similarity between the image and scene.
            similarity = torch.nn.functional.cosine_similarity(
                image_embedding,
                scene_embedding,
                dim=-1
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_scene = scene_id

        return best_scene, best_similarity

    @staticmethod
    def _load_image(frame_path: Path):
        """
        Load an image from disk as a PIL RGB image.

        If an image is corrupted or truncated, return None
        so that one bad frame does not stop the entire dataset scan.
        """

        from PIL import Image

        try:
            # Open the image and convert it to RGB.
            image = Image.open(frame_path)
            image.load()

            return image.convert("RGB")

        except (OSError, ValueError) as e:

            # Report the problematic frame.
            print(
                f"Skipping corrupted image: {frame_path.name}"
            )
            print(f"Reason: {e}")

            return None