from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class ByteTrackTracker:
    """
    ByteTrack-based multi-object tracker.

    Responsibilities
    ----------------
    1. Load a YOLO detection model.
    2. Detect objects in consecutive frames.
    3. Assign persistent tracking IDs using ByteTrack.
    4. Create annotated frames with bounding boxes and IDs.
    5. Save annotated frames.
    6. Return structured tracking information.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Load YOLO model
        self.model = YOLO(model_name)

    def track_frames(
        self,
        frames_dir: Path,
        output_dir: Path,
        max_frames: int | None = None,
    ):
        """
        Track objects across a sequence of frames.

        Parameters
        ----------
        frames_dir : Path
            Folder containing input JPG frames.

        output_dir : Path
            Folder where annotated frames will be saved.

        max_frames : int | None
            Optional limit for testing.
            None means process all frames.

        Returns
        -------
        list
            Structured tracking information for all processed frames.
        """

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all JPG frames
        frame_paths = sorted(frames_dir.glob("*.jpg"))

        # Optional frame limit for testing
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        if not frame_paths:
            print("No JPG frames found.")
            return []

        print(f"Tracking {len(frame_paths)} frames...")
        print("=" * 70)

        # This will store ALL tracking information
        tracking_data = []

        # ---------------------------------------------------------
        # Process frames sequentially
        # ---------------------------------------------------------

        for frame_number, frame_path in enumerate(
            frame_paths,
            start=1,
        ):

            # Run YOLO + ByteTrack
            results = self.model.track(
                source=str(frame_path),
                conf=self.confidence_threshold,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            result = results[0]

            print(f"\nFrame {frame_number}: {frame_path.name}")
            print("-" * 70)

            # -----------------------------------------------------
            # Information for this particular frame
            # -----------------------------------------------------

            frame_tracks = []

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes

                # Bounding boxes
                xyxy = boxes.xyxy.cpu().tolist()

                # Class IDs
                class_ids = boxes.cls.cpu().tolist()

                # Confidence scores
                confidences = boxes.conf.cpu().tolist()

                # Tracking IDs
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().tolist()
                else:
                    track_ids = [None] * len(boxes)

                # -------------------------------------------------
                # Extract each tracked object
                # -------------------------------------------------

                for bbox, class_id, confidence, track_id in zip(
                    xyxy,
                    class_ids,
                    confidences,
                    track_ids,
                ):

                    class_id = int(class_id)

                    class_name = result.names[class_id]

                    if track_id is not None:
                        track_id = int(track_id)

                    # Structured information for this object
                    track = {
                        "frame_number": frame_number,
                        "frame_name": frame_path.name,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                    }

                    frame_tracks.append(track)
                    tracking_data.append(track)

                    # Terminal output
                    print(
                        f"ID={track_id} | "
                        f"class={class_name} | "
                        f"confidence={confidence:.2f} | "
                        f"bbox={bbox}"
                    )

            else:
                print("No objects detected.")

            # -----------------------------------------------------
            # Create annotated image
            # -----------------------------------------------------

            annotated_frame = result.plot()

            # YOLO returns BGR.
            # PIL expects RGB.
            annotated_image = Image.fromarray(
                annotated_frame[:, :, ::-1]
            )

            # -----------------------------------------------------
            # Save annotated image
            # -----------------------------------------------------

            output_path = output_dir / frame_path.name

            annotated_image.save(output_path)

        # ---------------------------------------------------------
        # Finished
        # ---------------------------------------------------------

        print("\n" + "=" * 70)
        print("Tracking completed.")

        print(f"Processed frames: {len(frame_paths)}")

        print(f"Total tracked detections: {len(tracking_data)}")

        print("Annotated frames saved to:")
        print(output_dir)

        return tracking_data