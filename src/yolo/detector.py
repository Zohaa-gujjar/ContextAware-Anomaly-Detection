from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around a pretrained YOLO object detection model.

    Responsibilities
    ----------------
    1. Load the pretrained YOLO model.
    2. Run object detection on an image.
    3. Extract detected object information.
    4. Save annotated detection images.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
    ):
        """
        Initialize the YOLO detector.

        Parameters
        ----------
        model_name : str
            Name of the pretrained YOLO model.

        confidence_threshold : float
            Minimum confidence required for a detection.
        """

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Load the pretrained YOLO model.
        self.model = YOLO(model_name)

    def detect(self, image):
        """
        Detect objects in an image.

        Parameters
        ----------
        image : PIL.Image.Image or image path
            Input image.

        Returns
        -------
        list
            List of detected objects.
        """

        # Run YOLO inference.
        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        # YOLO returns a list of result objects.
        result = results[0]

        # Extract detected bounding boxes.
        for box in result.boxes:

            # Predicted class ID.
            class_id = int(box.cls[0].item())

            # Human-readable object name.
            class_name = result.names[class_id]

            # Confidence score.
            confidence = float(box.conf[0].item())

            # Bounding box coordinates.
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    def detect_from_path(self, image_path: Path):
        """
        Detect objects directly from an image file.

        Parameters
        ----------
        image_path : Path
            Path to the input image.

        Returns
        -------
        list
            List of detected objects.
        """

        # Open image as RGB.
        image = Image.open(image_path).convert("RGB")

        # Run detection.
        return self.detect(image)

    def detect_and_save(
        self,
        image_path: Path,
        output_path: Path,
    ):
        """
        Run YOLO detection and save the image
        with bounding boxes, class names,
        and confidence scores.
        """

        # Run YOLO directly on the image path.
        results = self.model(
            str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        result = results[0]

        # Create annotated image.
        annotated_image = result.plot()

        # Create output directory if necessary.
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save annotated image.
        Image.fromarray(
            annotated_image[:, :, ::-1]
        ).save(output_path)

        # Also return structured detections.
        return self.detect_from_path(image_path)