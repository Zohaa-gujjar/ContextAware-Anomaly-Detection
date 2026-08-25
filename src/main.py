from pathlib import Path

from PIL import Image

from src.clip.clip_model import CLIPModelWrapper
from src.clip.scene_classifier import SceneClassifier
from src.scene.scene_catalog import SceneCatalog
from src.Tracking.tracker import ByteTrackTracker
from src.context.context_builder import ContextBuilder
from src.prompting.dy_prompt_generator import DynamicPromptGenerator


def main():

    print("=" * 70)
    print("FULL CONTEXT-AWARE PIPELINE")
    print("=" * 70)

    # =========================================================
    # 1. PATHS
    # =========================================================

    # ShanghaiTech training dataset
    dataset_path = Path(
        r"C:\Users\Zoi\Downloads\shanghaitech_EDA\dataset\SHANGHAI_TRAIN"
    )

    # Sequence we want to process
    frames_dir = Path(
        r"C:\Users\Zoi\Downloads\shanghaitech_EDA\dataset\SHANGHAI_TRAIN\frames\11_0176"
    )

    # Scene knowledge catalog
    scene_catalog_path = Path(
        "Knowledge/scene_catalog.yaml"
    )

    # Output directory for annotated tracking frames
    output_dir = Path(
        "outputs/tracking/11_0176"
    )

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist:\n{dataset_path}"
        )

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frame sequence does not exist:\n{frames_dir}"
        )

    if not scene_catalog_path.exists():
        raise FileNotFoundError(
            f"Scene catalog does not exist:\n{scene_catalog_path}"
        )

    # =========================================================
    # 2. LOAD CLIP
    # =========================================================

    print("\n[1/6] Loading CLIP...")

    clip_model = CLIPModelWrapper()

    print("CLIP loaded.")

    # =========================================================
    # 3. CREATE SCENE CLASSIFIER
    # =========================================================

    print("\n[2/6] Building scene embeddings...")

    scene_classifier = SceneClassifier(
        clip_model=clip_model
    )

    scene_classifier.build_scene_embeddings(
        dataset_path=dataset_path,
        samples_per_sequence=5,
    )

    print(
        f"\nScene embeddings created for "
        f"{len(scene_classifier.scene_embeddings)} scenes."
    )

    # =========================================================
    # 4. LOAD SCENE CATALOG
    # =========================================================

    print("\n[3/6] Loading Scene Catalog...")

    scene_catalog = SceneCatalog.from_yaml(
        scene_catalog_path
    )

    print(scene_catalog)

    # =========================================================
    # 5. GET A REAL FRAME FOR SCENE CLASSIFICATION
    # =========================================================

    frame_paths = sorted(
        frames_dir.glob("*.jpg")
    )

    if not frame_paths:
        raise RuntimeError(
            f"No JPG frames found in:\n{frames_dir}"
        )

    # Use the first real frame of the sequence.
    scene_test_frame = frame_paths[0]

    print(
        f"\nScene classification frame: "
        f"{scene_test_frame.name}"
    )

    image = Image.open(
        scene_test_frame
    ).convert("RGB")

    # =========================================================
    # 6. CLIP → SCENE ID
    # =========================================================

    print("\n[4/6] Classifying scene with CLIP...")

    predicted_scene, scene_similarity = (
        scene_classifier.predict(image)
    )

    print(
        f"Predicted Scene ID : {predicted_scene}"
    )

    print(
        f"Scene Similarity    : "
        f"{scene_similarity:.4f}"
    )

    # =========================================================
    # 7. SCENE ID → SCENE KNOWLEDGE
    # =========================================================

    print("\nRetrieving scene knowledge...")

    scene = scene_catalog.get_scene(
        predicted_scene
    )

    # Convert Scene object into the dictionary
    # expected by ContextBuilder.
    scene_info = scene.info()

    print(
        f"Scene Name          : "
        f"{scene.scene_name}"
    )

    print(
        f"Scene Type          : "
        f"{scene.scene_type}"
    )

    print(
        f"Environment         : "
        f"{scene.indoor_outdoor}"
    )

    print(
        f"Common Objects      : "
        f"{scene.common_objects}"
    )

    print(
        f"Normal Activities   : "
        f"{scene.normal_activities}"
    )

    print(
        f"Contextual Concerns : "
        f"{scene.contextual_concerns}"
    )

    # =========================================================
    # 8. YOLO + BYTETRACK
    # =========================================================

    print("\n[5/6] Running YOLO + ByteTrack...")

    tracker = ByteTrackTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
    )

    # max_frames=None means:
    # process ALL frames in the selected sequence.
    tracking_data = tracker.track_frames(
        frames_dir=frames_dir,
        output_dir=output_dir,
        max_frames=None,
    )

    print(
        f"\nTotal tracking records returned: "
        f"{len(tracking_data)}"
    )

    # =========================================================
    # 9. BUILD CONTEXT
    # =========================================================

    print("\nBuilding Context...")

    context_builder = ContextBuilder()

    context = context_builder.build_context(
        scene_id=predicted_scene,
        scene_similarity=scene_similarity,
        scene_info=scene_info,
        tracking_data=tracking_data,
    )

    # =========================================================
    # 10. GENERATE DYNAMIC PROMPT
    # =========================================================

    print("\nGenerating Dynamic Prompt...")

    prompt_generator = DynamicPromptGenerator()

    generated_prompt = (
        prompt_generator.generate_prompt(
            context
        )
    )

    # =========================================================
    # 11. DISPLAY FINAL CONTEXT
    # =========================================================

    print("\n")
    print("=" * 70)
    print("FINAL INTEGRATED CONTEXT")
    print("=" * 70)

    print("\nScene:")
    print(context["scene"])

    print("\nObjects:")
    print(context["objects"])

    print("\nTracking:")
    print(context["tracking"])

    # =========================================================
    # 12. DISPLAY GENERATED PROMPT
    # =========================================================

    print("\n")
    print("=" * 70)
    print("GENERATED DYNAMIC PROMPT")
    print("=" * 70)

    print(generated_prompt)

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()