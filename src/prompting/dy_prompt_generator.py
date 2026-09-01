class DynamicPromptGenerator:
    """
    Generates a compact, context-aware reasoning prompt from
    ContextBuilder schema v1.0.

    Responsibilities
    ----------------
    1. Receive the structured context produced by ContextBuilder.
    2. Extract scene, domain-knowledge, object, and high-level
       tracking information.
    3. Construct a compact task-specific prompt for Qwen2.5-VL.

    This module does NOT:
    - perform object detection,
    - perform tracking,
    - classify the scene,
    - decide whether an anomaly actually occurred,
    - communicate with Qwen.

    Raw tracking coordinates and repetitive track-level metadata
    are intentionally not included in the Qwen prompt. The
    ContextBuilder still retains that information.
    """

    SUPPORTED_SCHEMA_VERSION = "1.0"

    def __init__(self):
        """
        Initialize the Dynamic Prompt Generator.
        """
        pass

    # -------------------------------------------------------------
    # Tracking summarization
    # -------------------------------------------------------------

    def _summarize_tracking(self, tracked_objects):
        """
        Convert detailed track-level information into a compact,
        factual summary suitable for Qwen.

        The raw ContextBuilder information is preserved, but
        unnecessary numerical details such as bounding boxes and
        individual confidence values are not passed to Qwen.

        Parameters
        ----------
        tracked_objects : list
            Track-level summaries from ContextBuilder v1.0.

        Returns
        -------
        list
            Compact factual tracking observations.
        """

        if not tracked_objects:
            return [
                "No individual tracking information is available."
            ]

        observations = []

        # ---------------------------------------------------------
        # Overall persistence information
        # ---------------------------------------------------------

        long_tracks = []
        short_tracks = []

        for obj in tracked_objects:

            frames_visible = obj.get(
                "frames_visible"
            )

            if isinstance(frames_visible, (int, float)):

                if frames_visible >= 30:
                    long_tracks.append(obj)

                else:
                    short_tracks.append(obj)

        if long_tracks:
            observations.append(
                f"{len(long_tracks)} tracked objects remained "
                f"visible for at least 30 frames."
            )

        if short_tracks:
            observations.append(
                f"{len(short_tracks)} tracked objects were visible "
                f"for fewer than 30 frames."
            )

        # ---------------------------------------------------------
        # Class-level tracking presence
        # ---------------------------------------------------------

        class_track_counts = {}

        for obj in tracked_objects:

            class_name = obj.get(
                "class_name",
                "Unknown"
            )

            class_track_counts[class_name] = (
                class_track_counts.get(class_name, 0) + 1
            )

        if class_track_counts:

            sorted_classes = sorted(
                class_track_counts.items(),
                key=lambda x: (-x[1], x[0])
            )

            for class_name, count in sorted_classes:

                observations.append(
                    f"{count} tracked instances of "
                    f"{class_name} were observed."
                )

        return observations

    # -------------------------------------------------------------
    # Main prompt generation
    # -------------------------------------------------------------

    def generate_prompt(self, context):
        """
        Generate a compact reasoning prompt from
        ContextBuilder v1.0.

        Parameters
        ----------
        context : dict
            Structured context produced by ContextBuilder.

        Returns
        -------
        str
            Compact dynamic reasoning prompt for Qwen2.5-VL.
        """

        # =========================================================
        # 1. Validate ContextBuilder schema
        # =========================================================

        if not isinstance(context, dict):
            raise TypeError(
                "Context must be a dictionary."
            )

        schema_version = context.get(
            "schema_version"
        )

        if schema_version != self.SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ContextBuilder schema version: "
                f"{schema_version}. "
                f"Expected: {self.SUPPORTED_SCHEMA_VERSION}."
            )

        # =========================================================
        # 2. Extract scene information
        # =========================================================

        scene = context.get(
            "scene",
            {}
        )

        scene_id = scene.get(
            "scene_id",
            "Unknown"
        )

        similarity = scene.get(
            "similarity",
            "Unknown"
        )

        knowledge = scene.get(
            "knowledge",
            {}
        )

        scene_name = knowledge.get(
            "scene_name",
            "Unknown"
        )

        scene_type = knowledge.get(
            "scene_type",
            "Unknown"
        )

        normal_activities = knowledge.get(
            "normal_activities",
            []
        )

        contextual_concerns = knowledge.get(
            "contextual_concerns",
            []
        )

        scene_purpose = knowledge.get(
            "scene_purpose",
            []
        )

        environmental_constraints = knowledge.get(
            "environmental_constraints",
            []
        )

        # =========================================================
        # 3. Extract object information
        # =========================================================

        objects = context.get(
            "objects",
            {}
        )

        object_counts = objects.get(
            "object_counts",
            {}
        )

        tracked_objects = objects.get(
            "tracked_objects",
            []
        )

        # =========================================================
        # 4. Extract tracking information
        # =========================================================

        tracking = context.get(
            "tracking",
            {}
        )

        frames_processed = tracking.get(
            "frames_processed",
            0
        )

        total_records = tracking.get(
            "total_records",
            0
        )

        unique_track_ids = tracking.get(
            "unique_track_ids",
            0
        )

        # =========================================================
        # 5. Format scene knowledge
        # =========================================================

        normal_activity_text = (
            ", ".join(normal_activities)
            if normal_activities
            else "Not specified."
        )

        concern_text = (
            ", ".join(contextual_concerns)
            if contextual_concerns
            else "Not specified."
        )

        purpose_text = (
            ", ".join(scene_purpose)
            if scene_purpose
            else "Not specified."
        )

        constraint_text = (
            ", ".join(environmental_constraints)
            if environmental_constraints
            else "Not specified."
        )

        # =========================================================
        # 6. Format object counts
        # =========================================================

        if object_counts:

            object_lines = []

            for class_name, count in sorted(
                object_counts.items()
            ):
                object_lines.append(
                    f"- {class_name}: {count}"
                )

            object_summary = "\n".join(
                object_lines
            )

        else:

            object_summary = (
                "- No tracked objects detected."
            )

        # =========================================================
        # 7. Create compact tracking summary
        # =========================================================

        tracking_observations = self._summarize_tracking(
            tracked_objects
        )

        tracking_summary = "\n".join(
            f"- {observation}"
            for observation in tracking_observations
        )

        # =========================================================
        # 8. Construct compact reasoning prompt
        # =========================================================

        prompt = f"""
You are a visual reasoning system analyzing a surveillance
video sequence.

Your task is to determine whether the activity visible in the
provided frames is consistent with the normal activity expected
in the identified scene or whether there is evidence of anomalous
activity.

The contextual concerns below represent possible deviations that
are relevant to the scene. Their presence does NOT mean that the
corresponding event is occurring.

Use the visual evidence as the primary basis for your conclusion.
Use the supplied contextual information to interpret that evidence.

============================================================
SCENE CONTEXT
============================================================

Scene ID:
{scene_id}

Scene Name:
{scene_name}

Scene Type:
{scene_type}

Scene Classification Similarity:
{similarity}

Scene Purpose:
{purpose_text}

Environmental Constraints:
{constraint_text}

============================================================
EXPECTED SCENE CONTEXT
============================================================

Normal Activities:
{normal_activity_text}

Contextual Concerns:
{concern_text}

============================================================
OBSERVED CONTEXT
============================================================

Tracked Objects by Class:
{object_summary}

Frames Processed:
{frames_processed}

Unique Tracked Objects:
{unique_track_ids}

Tracking Summary:
{tracking_summary}

============================================================
REASONING TASK
============================================================

Examine all provided frames carefully.

1. Identify the main activity or activities visible in the
   sequence.

2. Compare the observed activity with the normal activities
   expected for this scene.

3. Consider the contextual concerns as possible deviations,
   but do not assume that any concern is actually occurring.

4. Determine whether the observed activity is:

   - NORMAL
   - ANOMALOUS
   - UNCERTAIN

5. An anomalous classification must be supported by observable
   visual evidence.

6. If anomalous, briefly describe the observable behavior that
   supports the conclusion.

7. If the visual evidence is insufficient, classify the result
   as UNCERTAIN rather than assuming an anomaly.

============================================================
RESPONSE FORMAT
============================================================

Classification:
[NORMAL / ANOMALOUS / UNCERTAIN]

Observed Activity:
[Brief description of what is visibly happening]

Reasoning:
[Brief explanation comparing the observed activity with the
expected activity for the scene]

Evidence:
[Specific visual observations supporting the classification]

Do not infer events that cannot be supported by the visual
evidence.
""".strip()

        return prompt