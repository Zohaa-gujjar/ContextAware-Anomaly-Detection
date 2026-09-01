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
        # 8. Construct context-aware reasoning prompt
        # =========================================================

        prompt = f"""
You are a visual reasoning system analyzing a surveillance
video sequence.

Your task is to determine whether the activity visible in the
provided frames is NORMAL, ANOMALOUS, or UNCERTAIN for the
identified scene.

Use the visual evidence as the PRIMARY basis for your conclusion.
Use the supplied scene context only to interpret that evidence.

The contextual concerns are POSSIBLE deviations, not evidence
that those events are occurring.

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
ANOMALY DECISION RULES
============================================================

1. Identify ONLY actions or events that are visibly supported
   by the provided frames.

2. Objects are NOT anomalies by themselves.
   A backpack, handbag, bicycle, luggage, or other object does
   not constitute anomalous behavior.

3. Do NOT classify something as anomalous merely because it is:
   - uncommon,
   - unusual,
   - unexpected,
   - carried by a person,
   - associated with a single person,
   - or different from other people in the scene.

4. Compare the OBSERVED BEHAVIOR with the listed normal
   activities for the scene.

5. ANOMALOUS requires a specific observable behavior or event
   that clearly deviates from the expected activity.

6. Do NOT infer intentions, suspiciousness, hidden events,
   gatherings, or circumstances that are not visibly supported.

7. If the visible activity is consistent with a normal activity
   and no specific anomalous behavior is visible, classify it
   as NORMAL.

8. If the frames do not provide enough evidence to determine
   whether an anomaly is occurring, classify it as UNCERTAIN.

9. Before assigning ANOMALOUS, identify the exact observable
   behavior that makes the activity anomalous.

============================================================
REASONING TASK
============================================================

Examine all provided frames together.

1. Identify the main visible activity.

2. Compare that activity with the normal activities expected
   for this scene.

3. Check whether any specific anomalous behavior is visibly
   occurring.

4. Do not treat contextual concerns or detected objects as
   evidence of an anomaly unless the corresponding behavior
   is actually visible.

5. Determine the classification:

   NORMAL
   ANOMALOUS
   UNCERTAIN

============================================================
RESPONSE FORMAT
============================================================

Classification:
[NORMAL / ANOMALOUS / UNCERTAIN]

Observed Activity:
[Brief description of what is visibly happening]

Reasoning:
[Brief comparison between the observed activity and the
expected activity for the scene]

Evidence:
[Specific visual observations supporting the classification]

If ANOMALOUS, explicitly state the specific visible behavior
that makes it anomalous.

Do not infer events that cannot be supported by the visual
evidence.
""".strip()

        return prompt