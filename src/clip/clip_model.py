from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch


class CLIPModelWrapper:
    """
    Wrapper around Hugging Face's pretrained CLIP model.

    Responsibilities
    ----------------
    1. Load the pretrained CLIP model.
    2. Load the corresponding image processor.
    3. Extract final image embeddings.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize the CLIP model.

        Parameters
        ----------
        model_name : str
            Name of the pretrained CLIP model.
        """

        self.model_name = model_name

        # Load the pretrained CLIP model.
        self.model = CLIPModel.from_pretrained(model_name)

        # Load the processor used to prepare images for CLIP.
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image: Image.Image):
        """
        Extract a normalized CLIP image embedding.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image.

        Returns
        -------
        torch.Tensor
            Normalized 512-dimensional image embedding.
        """

        # Convert the PIL image into the format expected by CLIP.
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        # We are only extracting features, so gradients are unnecessary.
        with torch.no_grad():

            # Pass the image through CLIP's vision encoder.
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            # Get the pooled representation of the complete image.
            pooled_output = vision_outputs.pooler_output

            # Project the pooled representation into CLIP's
            # final image embedding space.
            embedding = self.model.visual_projection(
                pooled_output
            )

        # Normalize the final image embedding.
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        return embedding