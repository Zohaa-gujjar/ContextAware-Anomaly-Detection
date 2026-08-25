class DynamicPromptGenerator:
    """
    Generates a dynamic, context-aware reasoning prompt from
    ContextBuilder schema v1.0.

    Responsibilities
    ----------------
    1. Receive the structured context produced by ContextBuilder.
    2. Extract scene, domain-knowledge, object, and tracking information.
    3. Construct a task-specific prompt for Qwen2.5-VL.

    This module does NOT:
    - perform object detection,
    - perform tracking,
    - classify the scene,
    - decide whether an anomaly actually occurred,
    - communicate with Qwen.

    It only constructs the reasoning prompt.
    """

    SUPPORTED_SCHEMA_VERSION = "1.0"

    def __init__(self):
        """
        Initialize the Dynamic Prompt Generator.
        """
        pass

    def generate_prompt(self, context):
        """
        Generate a reasoning prompt from ContextBuilder v1.0.

        Parameters
        ----------
        context : dict
            Structured context produced by ContextBuilder.

        Returns
        -------
        str
            Dynamic reasoning prompt for Qwen2.5-VL.
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
        # 7. Format tracked-object information
        # =========================================================

        if tracked_objects:

            track_lines = []

            for obj in tracked_objects:

                track_id = obj.get(
                    "track_id",
                    "Unknown"
                )

                class_name = obj.get(
                    "class_name",
                    "Unknown"
                )

                first_frame = obj.get(
                    "first_seen_frame",
                    "Unknown"
                )

                last_frame = obj.get(
                    "last_seen_frame",
                    "Unknown"
                )

                frames_visible = obj.get(
                    "frames_visible",
                    "Unknown"
                )

                average_confidence = obj.get(
                    "average_confidence",
                    "Unknown"
                )

                first_bbox = obj.get(
                    "first_bbox",
                    "Unknown"
                )

                last_bbox = obj.get(
                    "last_bbox",
                    "Unknown"
                )

                track_lines.append(
                    (
                        f"- Track {track_id}: "
                        f"{class_name}; "
                        f"frames {first_frame}-{last_frame}; "
                        f"visible for {frames_visible} frames; "
                        f"average detection confidence "
                        f"{average_confidence}; "
                        f"initial bounding box "
                        f"{first_bbox}; "
                        f"final bounding box "
                        f"{last_bbox}"
                    )
                )

            tracking_object_text = "\n".join(
                track_lines
            )

        else:

            tracking_object_text = (
                "- No individual tracking information available."
            )

        # =========================================================
        # 8. Construct the reasoning prompt
        # =========================================================

        prompt = f"""
You are a visual reasoning system analyzing a surveillance
video sequence.

Your task is to determine whether the activity occurring in
the provided sequence is consistent with the normal activity
expected in the identified scene or whether there is evidence
of anomalous activity.

IMPORTANT:
The contextual concerns listed below are possible types of
deviation that are relevant to this scene. Their presence in
the list does NOT mean that the corresponding event is
actually occurring.

You must base your conclusion on the visual evidence in the
provided video/frames.

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
OBSERVED OBJECT CONTEXT
============================================================

Tracked Objects by Class:
{object_summary}

Number of Frames Processed:
{frames_processed}

Total Tracking Records:
{total_records}

Unique Tracked Objects:
{unique_track_ids}


============================================================
TRACK-LEVEL OBSERVATIONS
============================================================

{tracking_object_text}


============================================================
REASONING TASK
============================================================

Examine the provided visual sequence carefully.

1. Identify the main activity or activities occurring in
   the sequence.

2. Compare the observed activity with the normal activities
   expected for this scene.

3. Consider the contextual concerns as possible deviations
   that deserve attention.

4. Determine whether the observed activity is:

   - NORMAL
   - ANOMALOUS
   - UNCERTAIN

5. Do not classify an event as anomalous merely because a
   contextual concern is listed. An anomaly must be supported
   by observable visual evidence.

6. If you determine that the activity is anomalous, describe
   the observable behavior that supports that conclusion.

7. If the evidence is insufficient to make a reliable
   determination, explicitly state that the result is
   uncertain rather than assuming an anomaly.

============================================================
RESPONSE FORMAT
============================================================

Classification:
[NORMAL / ANOMALOUS / UNCERTAIN]

Observed Activity:
[Brief description of what is visibly happening]

Reasoning:
[Explain why the observed activity is or is not consistent
with the expected activity for this scene.]

Evidence:
[List the specific visual observations supporting the
classification.]

Do not infer events that cannot be supported by the visual
evidence.
""".strip()

        return prompt