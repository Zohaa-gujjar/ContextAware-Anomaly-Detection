class ContextBuilder:
    """
    Builds a structured representation of the current surveillance context.

    The Context Builder combines:
    1. Scene information from the scene knowledge base.
    2. Detected/tracked objects from YOLO + ByteTrack.
    3. Scene classification information from CLIP.

    This module does NOT decide whether an event is anomalous.
    Its purpose is to organize the available information into
    a clean, stable contextual representation for downstream reasoning.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self):
        """
        Initialize the Context Builder.
        """
        pass

    # -------------------------------------------------------------
    # Track-level summarization
    # -------------------------------------------------------------

    def _summarize_tracks(self, tracking_data):
        """
        Convert frame-level tracking records into track-level summaries.

        Instead of passing every bounding box from every frame
        downstream, this method summarizes each unique tracked
        object.

        Parameters
        ----------
        tracking_data : list
            Complete YOLO + ByteTrack tracking records.

        Returns
        -------
        list
            One summarized record for each unique track ID.
        """

        tracks = {}

        for record in tracking_data:

            track_id = record.get("track_id")

            if track_id is None:
                continue

            class_name = record.get("class_name")

            frame_number = record.get("frame_number")

            confidence = record.get("confidence")

            bbox = record.get("bbox")

            # -----------------------------------------------------
            # Create a new track
            # -----------------------------------------------------

            if track_id not in tracks:

                tracks[track_id] = {
                    "track_id": track_id,
                    "class_name": class_name,

                    "first_seen_frame": frame_number,
                    "last_seen_frame": frame_number,

                    "frames_visible": 0,

                    "confidence_sum": 0.0,

                    "first_bbox": bbox,
                    "last_bbox": bbox,
                }

            track = tracks[track_id]

            # -----------------------------------------------------
            # Update frame information
            # -----------------------------------------------------

            if frame_number is not None:

                if (
                    track["first_seen_frame"] is None
                    or frame_number < track["first_seen_frame"]
                ):
                    track["first_seen_frame"] = frame_number

                if (
                    track["last_seen_frame"] is None
                    or frame_number > track["last_seen_frame"]
                ):
                    track["last_seen_frame"] = frame_number

            # -----------------------------------------------------
            # Update visibility count
            # -----------------------------------------------------

            track["frames_visible"] += 1

            # -----------------------------------------------------
            # Update confidence
            # -----------------------------------------------------

            if confidence is not None:

                track["confidence_sum"] += float(confidence)

            # -----------------------------------------------------
            # Update latest bounding box
            # -----------------------------------------------------

            if bbox is not None:

                track["last_bbox"] = bbox

        # ---------------------------------------------------------
        # Finalize track summaries
        # ---------------------------------------------------------

        summarized_tracks = []

        for track in tracks.values():

            frames_visible = track["frames_visible"]

            if frames_visible > 0:

                average_confidence = (
                    track["confidence_sum"] / frames_visible
                )

            else:

                average_confidence = 0.0

            summarized_track = {
                "track_id": track["track_id"],
                "class_name": track["class_name"],

                "first_seen_frame": track["first_seen_frame"],
                "last_seen_frame": track["last_seen_frame"],

                "frames_visible": frames_visible,

                "average_confidence": round(
                    average_confidence,
                    4,
                ),

                "first_bbox": track["first_bbox"],
                "last_bbox": track["last_bbox"],
            }

            summarized_tracks.append(summarized_track)

        # Sort by track ID for deterministic output
        summarized_tracks.sort(
            key=lambda x: x["track_id"]
        )

        return summarized_tracks

    # -------------------------------------------------------------
    # Main context builder
    # -------------------------------------------------------------

    def build_context(
        self,
        scene_id,
        scene_similarity,
        scene_info,
        tracking_data,
    ):
        """
        Build a structured surveillance context.

        Parameters
        ----------
        scene_id : int
            Scene predicted by CLIP.

        scene_similarity : float
            Similarity score produced by CLIP.

        scene_info : dict
            Knowledge associated with the predicted scene.

        tracking_data : list
            Complete YOLO + ByteTrack tracking records.

        Returns
        -------
        dict
            Structured surveillance context following schema v1.0.
        """

        # =========================================================
        # 1. Summarize tracked objects
        # =========================================================

        tracked_objects = self._summarize_tracks(
            tracking_data
        )

        # =========================================================
        # 2. Count unique objects by class
        # =========================================================

        object_counts = {}

        for obj in tracked_objects:

            class_name = obj["class_name"]

            object_counts[class_name] = (
                object_counts.get(class_name, 0) + 1
            )

        # =========================================================
        # 3. Determine number of processed frames
        # =========================================================

        frame_numbers = set()

        for record in tracking_data:

            frame_number = record.get("frame_number")

            if frame_number is not None:

                frame_numbers.add(frame_number)

        frames_processed = len(frame_numbers)

        # =========================================================
        # 4. Build final stable context
        # =========================================================

        context = {

            # -----------------------------------------------------
            # Schema information
            # -----------------------------------------------------

            "schema_version": self.SCHEMA_VERSION,

            # -----------------------------------------------------
            # Scene information
            # -----------------------------------------------------

            "scene": {
                "scene_id": scene_id,

                "similarity": round(
                    float(scene_similarity),
                    4,
                ),

                "knowledge": scene_info,
            },

            # -----------------------------------------------------
            # Object information
            # -----------------------------------------------------

            "objects": {

                "object_counts": object_counts,

                "tracked_objects": tracked_objects,
            },

            # -----------------------------------------------------
            # Tracking information
            # -----------------------------------------------------

            "tracking": {

                "frames_processed": frames_processed,

                "total_records": len(
                    tracking_data
                ),

                "unique_track_ids": len(
                    tracked_objects
                ),
            },
        }

        return context