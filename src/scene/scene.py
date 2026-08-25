
from dataclasses import dataclass
from typing import List


@dataclass
class Scene:
    """
    Represents a single scene described in the scene catalog.

    A Scene object stores all contextual knowledge about one
    surveillance environment.

    Example:
        Scene 01 -> Campus Walkway
        Scene 02 -> Road Outside Cafe
        ...
    """

    # -------------------------------------------------------
    # Basic Information
    # -------------------------------------------------------

    scene_id: int
    scene_name: str

    # -------------------------------------------------------
    # Environment Information
    # -------------------------------------------------------

    indoor_outdoor: str
    scene_type: str
    description: str

    # -------------------------------------------------------
    # Camera Information
    # -------------------------------------------------------

    camera_type: str
    camera_angle: str
    field_of_view: str

    # -------------------------------------------------------
    # Scene Knowledge
    # -------------------------------------------------------

    reference_sequences: List[str]

    common_objects: List[str]

    normal_activities: List[str]

    contextual_concerns: List[str]

    environmental_constraints: List[str]

    scene_purpose: List[str]

    # -------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------

    def info(self):
        """
        Returns all scene information in dictionary format.

        Useful for debugging, printing, and future JSON export.
        """

        return {
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "scene_type": self.scene_type,
            "camera": self.camera_type,
            "reference_sequences": self.reference_sequences,
            "common_objects": self.common_objects,
            "normal_activities": self.normal_activities,
            "contextual_concerns": self.contextual_concerns,
            "environmental_constraints": self.environmental_constraints,
            "scene_purpose": self.scene_purpose,
        }

    @property
    def total_reference_sequences(self):
        """
        Returns the number of reference sequences
        available for this scene.
        """

        return len(self.reference_sequences)

    @property
    def total_common_objects(self):
        """
        Returns the number of commonly observed objects.
        """

        return len(self.common_objects)

    @property
    def total_normal_activities(self):
        """
        Returns the number of normal activities.
        """

        return len(self.normal_activities)

    @property
    def total_contextual_concerns(self):
        """
        Returns the number of contextual concerns.
        """

        return len(self.contextual_concerns)

    def __repr__(self):
        return (
            f"Scene("
            f"{self.scene_id}: "
            f"{self.scene_name}"
            f")"
        )