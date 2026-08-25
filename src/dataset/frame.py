from pathlib import Path
import cv2
from PIL import Image


class Frame:
    """
    Represents a single image frame.
    """

    def __init__(self, image_path: Path):
        self.image_path = image_path

    @property
    def filename(self):
        return self.image_path.name

    @property
    def frame_id(self):
        return self.image_path.stem

    def load(self):
        """
        Load the image using OpenCV.
        """
        return cv2.imread(str(self.image_path))

    def load_pil_image(self):
        """
        Load the frame as a PIL Image.

        Returns
        -------
        PIL.Image.Image
            Frame in RGB format.

        Notes
        -----
        OpenCV loads images in BGR format, while CLIP expects RGB.
        Therefore, we convert BGR to RGB before creating the PIL image.
        """

        # Read image using OpenCV
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        return Image.fromarray(image)

    def __repr__(self):
        return f"Frame({self.filename})"