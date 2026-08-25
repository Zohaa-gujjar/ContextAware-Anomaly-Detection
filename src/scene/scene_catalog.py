
from pathlib import Path
import yaml

from src.scene.scene import Scene


class SceneCatalog:
    """
    Represents the complete Scene Catalog.

    It loads all scene information from scene_catalog.yaml
    and converts every scene into a Scene object.
    """

    def __init__(self, yaml_path: Path):
        """
        Initialize the SceneCatalog.

        Parameters
        ----------
        yaml_path : Path
            Path to scene_catalog.yaml
        """

        self.yaml_path = yaml_path

        # Dictionary:
        # key = scene_id
        # value = Scene object
        self.scenes = {}

        # Automatically load the catalog
        self.load_catalog()

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """
        Factory method.

        Example
        -------
        catalog = SceneCatalog.from_yaml(path)
        """

        return cls(yaml_path)

    def load_catalog(self):
        """
        Load all scenes from the YAML catalog.

        This function:
        1. Opens the YAML file.
        2. Reads all scene information.
        3. Creates a Scene object for every scene.
        4. Stores all Scene objects in a dictionary.
        """

        # ----------------------------------------------------
        # Open and read the YAML file
        # ----------------------------------------------------
        with open(self.yaml_path, "r", encoding="utf-8") as file:
            catalog = yaml.safe_load(file)

        # ----------------------------------------------------
        # Create a Scene object for every scene in the catalog
        # ----------------------------------------------------
        for scene_key, scene_data in catalog.items():

            # Create one Scene object
            scene = Scene(

                # ----------------------------
                # Basic Information
                # ----------------------------
                scene_id=scene_data["scene_id"],
                scene_name=scene_data["scene_name"],

                # ----------------------------
                # Environment Information
                # ----------------------------
                indoor_outdoor=scene_data["environment"]["indoor_outdoor"],
                scene_type=scene_data["environment"]["scene_type"],
                description=scene_data["environment"]["description"],

                # ----------------------------
                # Camera Information
                # ----------------------------
                camera_type=scene_data["camera"]["type"],
                camera_angle=scene_data["camera"]["angle"],
                field_of_view=scene_data["camera"]["field_of_view"],

                # ----------------------------
                # Scene Knowledge
                # ----------------------------
                reference_sequences=scene_data["reference_sequences"],
                common_objects=scene_data["common_objects"],
                normal_activities=scene_data["normal_activities"],
                contextual_concerns=scene_data["contextual_concerns"],
                environmental_constraints=scene_data["environmental_constraints"],
                scene_purpose=scene_data["scene_purpose"],
            )

            # Store the Scene object using its scene ID
            self.scenes[scene.scene_id] = scene


    def get_scene(self, scene_id: int):
        """
        Return a Scene object using its scene ID.

        Parameters
        ----------
        scene_id : int
            ID of the scene to retrieve.
        """

        # Check whether the requested scene exists
        if scene_id not in self.scenes:
            raise ValueError(f"Scene {scene_id} does not exist in the catalog.")

        return self.scenes[scene_id]

    def info(self):
        """
        Return basic information about the Scene Catalog.
        """

        return {
            "yaml_path": str(self.yaml_path),
            "total_scenes": len(self.scenes),
            "scene_ids": sorted(self.scenes.keys()),
        }

    def __repr__(self):
        """
        Return a readable representation of the SceneCatalog object.
        """

        return f"SceneCatalog({len(self.scenes)} scenes)"