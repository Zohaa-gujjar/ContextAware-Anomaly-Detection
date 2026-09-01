# Context-Aware Anomaly Detection

A context-aware video anomaly detection research system that combines
scene understanding, object detection, multi-object tracking, structured
scene knowledge, dynamic prompt generation, and vision-language
reasoning.

## Research Question

> Can a surveillance system automatically understand the context of a
> scene, generate its own natural-language reasoning prompt, and use a
> Vision-Language Model to decide whether an event is anomalous?

The pipeline is:

`ShanghaiTech Video → CLIP Scene Recognition → YOLO11n Object Detection → ByteTrack Tracking → Scene/Domain Knowledge → ContextBuilder → Dynamic Prompt Generator → Qwen2.5-VL → Explanation/Decision`

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Goals](#goals)
-   [System Architecture](#system-architecture)
-   [Pipeline Components](#pipeline-components)
-   [Functionalities](#functionalities)
-   [Technology Stack](#technology-stack)
-   [Repository Structure](#repository-structure)
-   [Dataset](#dataset)
-   [Scene Knowledge](#scene-knowledge)
-   [ContextBuilder](#contextbuilder)
-   [Dynamic Prompt Generator](#dynamic-prompt-generator)
-   [Qwen2.5-VL](#qwen25-vl)
-   [Kaggle Setup](#kaggle-setup)
-   [Running the Pipeline on Kaggle](#running-the-pipeline-on-kaggle)
-   [Dynamic Prompt + Qwen Inference
    Code](#dynamic-prompt--qwen-inference-code)
-   [Reproducibility Notes](#reproducibility-notes)
-   [Expected Output](#expected-output)
-   [Current Scope](#current-scope)
-   [Future Work](#future-work)
-   [Acknowledgements](#acknowledgements)
-   [License](#license)

------------------------------------------------------------------------

## Overview

Traditional surveillance systems can detect objects, but object presence
alone does not determine whether an activity is appropriate for a
particular environment.

This project introduces an explicit context layer between
computer-vision perception and Vision-Language reasoning. The system
first identifies the scene, detects and tracks objects, retrieves
scene-specific knowledge, and organizes these observations into a
structured context representation. That representation is then converted
automatically into a natural-language reasoning prompt for Qwen2.5-VL.

The current implementation uses the ShanghaiTech Campus dataset for
research experimentation.

------------------------------------------------------------------------

## Goals

1.  **Scene-aware understanding** --- identify the environment
    represented by a surveillance sequence.
2.  **Object perception** --- detect relevant objects using YOLO11n.
3.  **Temporal tracking** --- associate detections across frames using
    ByteTrack.
4.  **Context construction** --- combine scene information, domain
    knowledge, objects, and tracking information.
5.  **Dynamic prompt generation** --- automatically construct a
    scene-conditioned reasoning prompt.
6.  **Multimodal reasoning** --- provide Qwen2.5-VL with both
    representative frames and generated context.
7.  **Explainable output** --- obtain a classification and
    natural-language reasoning that can be inspected by a researcher.

------------------------------------------------------------------------

## System Architecture

``` text
                    INPUT VIDEO / SEQUENCE
                              |
                              v
                     Frame Sampling
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
       CLIP Scene Recognition            YOLO11n Detection
             |                                 |
             |                                 v
             |                           ByteTrack
             |                                 |
             +----------------+----------------+
                              |
                              v
                     ContextBuilder
                              |
                              v
                  Structured Context v1.0
                              |
                              v
                DynamicPromptGenerator
                              |
                              v
                 Dynamic Reasoning Prompt
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          4 Representative          Natural-language
              Frames                    Context
                 |                         |
                 +------------+------------+
                              |
                              v
                  Qwen2.5-VL-3B-Instruct
                              |
                              v
                  Classification + Reasoning
```

------------------------------------------------------------------------

## Pipeline Components

### 1. Dataset Layer

The dataset layer provides Python abstractions for:

-   `Dataset`
-   `Sequence`
-   `Frame`
-   `Label`
-   dataset utility functions

ShanghaiTech sequence identifiers encode the scene and sequence, for
example:

``` text
13_007
```

where `13` is the scene identifier and `007` is the sequence identifier.

### 2. CLIP Scene Recognition

`SceneClassifier` builds representative CLIP embeddings for the 13
ShanghaiTech scenes and compares an input frame with these scene
representations using cosine similarity.

The scene classifier:

1.  Finds sequence folders.
2.  Groups sequences by scene ID.
3.  Samples frames.
4.  Generates CLIP image embeddings.
5.  Averages embeddings for each scene.
6.  Normalizes the scene representation.
7.  Predicts the best matching scene and similarity.

### 3. YOLO11n Object Detection

YOLO11n is loaded through Ultralytics and is used to identify objects in
surveillance frames.

Detection information includes:

-   object class
-   confidence
-   bounding box

### 4. ByteTrack Tracking

ByteTrack associates detections across consecutive frames and produces
track IDs.

The resulting tracking information can be summarized into:

-   total frames processed
-   total tracking records
-   unique track IDs
-   tracked-object class counts
-   persistence of tracked objects

### 5. Scene / Domain Knowledge

`Knowledge/scene_catalog.yaml` stores scene-specific information such
as:

-   scene name
-   scene type
-   normal activities
-   contextual concerns
-   scene purpose
-   environmental constraints

This gives the reasoning stage knowledge about what is expected in a
particular environment.

### 6. ContextBuilder

The ContextBuilder combines:

``` text
CLIP scene information
+
Scene Catalog knowledge
+
YOLO detections
+
ByteTrack tracking
```

into a structured context representation.

### 7. Dynamic Prompt Generator

`src/prompting/dy_prompt_generator.py` converts ContextBuilder schema
`1.0` into a natural-language prompt.

The generator intentionally summarizes tracking information instead of
sending repetitive bounding-box and confidence details directly to Qwen.

The generated prompt contains sections such as:

``` text
SCENE CONTEXT
EXPECTED SCENE CONTEXT
OBSERVED CONTEXT
ANOMALY DECISION RULES
REASONING TASK
```

### 8. Qwen2.5-VL

The final stage uses:

``` text
Qwen/Qwen2.5-VL-3B-Instruct
```

Qwen receives:

-   four representative frames
-   the generated dynamic reasoning prompt

The current response categories are:

``` text
NORMAL
ANOMALOUS
UNCERTAIN
```

together with an observed-activity description and reasoning.

------------------------------------------------------------------------

## Functionalities

-   ShanghaiTech sequence loading
-   Scene identification using CLIP
-   Object detection using YOLO11n
-   Multi-object tracking using ByteTrack
-   Scene-specific domain knowledge lookup
-   Structured contextual representation
-   Automatic dynamic prompt generation
-   Four-frame multimodal Qwen inference
-   Natural-language classification and reasoning
-   GPU execution through Kaggle

------------------------------------------------------------------------

## Technology Stack

  Area                       Technology
  -------------------------- ---------------------------
  Language                   Python
  Dataset                    ShanghaiTech Campus
  Scene Recognition          CLIP
  Object Detection           Ultralytics YOLO11n
  Tracking                   ByteTrack
  Vision-Language Model      Qwen2.5-VL-3B-Instruct
  Deep Learning              PyTorch
  Model Interface            Hugging Face Transformers
  Image Processing           Pillow
  Numerical Processing       NumPy
  Knowledge Representation   YAML
  Development                VS Code
  Version Control            Git / GitHub
  GPU Research Environment   Kaggle

------------------------------------------------------------------------

## Repository Structure

``` text
Context-Aware-Anomaly-Detection/
│
├── Knowledge/
│   └── scene_catalog.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── sequence.py
│   │   ├── frame.py
│   │   ├── label.py
│   │   └── utils.py
│   │
│   ├── scene/
│   │   ├── __init__.py
│   │   ├── scene.py
│   │   ├── scene_catalog.py
│   │   └── utils.py
│   │
│   ├── clip/
│   │   └── ...
│   │
│   ├── yolo/
│   │   └── ...
│   │
│   ├── Tracking/
│   │   └── ...
│   │
│   ├── context/
│   │   └── ...
│   │
│   ├── prompting/
│   │   └── dy_prompt_generator.py
│   │
│   ├── qwen/
│   │   └── qwen_model.py
│   │
│   └── main.py
│
├── .gitignore
├── README.md
└── ...
```

Large model-weight files should not be committed to Git. YOLO weights
can be downloaded by Ultralytics when required, while Qwen is downloaded
through Hugging Face.

------------------------------------------------------------------------

## Dataset

The current research dataset is the **ShanghaiTech Campus dataset**.

The repository does not include the dataset itself.

The expected training structure is:

``` text
SHANGHAI_TRAIN/
├── frames/
│   ├── 01_001/
│   ├── 01_002/
│   ├── ...
│   └── 13_.../
│
├── label/
│   ├── 01_001.npy
│   ├── 01_002.npy
│   └── ...
│
└── SHANGHAI_train.txt
```

Each sequence contains JPG frames and has a corresponding `.npy` label
file.

For Kaggle, the dataset should be added to the notebook through **Add
Input** rather than copied into the Git repository.

------------------------------------------------------------------------

## Scene Knowledge

The scene catalog is stored at:

``` text
Knowledge/scene_catalog.yaml
```

It provides contextual information for the ShanghaiTech scenes.

The information is used to answer questions such as:

-   What type of environment is this?
-   What activities are normal here?
-   What types of deviations are relevant?
-   What is the purpose of this environment?
-   Are there environmental constraints that affect interpretation?

The Scene Catalog is a local project component and does not require an
external API.

------------------------------------------------------------------------

## ContextBuilder

The current ContextBuilder uses schema version:

``` text
1.0
```

The conceptual structure is:

``` text
schema_version
│
├── scene
│   ├── scene_id
│   ├── similarity
│   └── knowledge
│
├── objects
│   ├── object_counts
│   └── tracked_objects
│
└── tracking
    ├── frames_processed
    ├── total_records
    └── unique_track_ids
```

The ContextBuilder is responsible for organizing information. The final
interpretation is performed by the Vision-Language reasoning stage.

------------------------------------------------------------------------

## Dynamic Prompt Generator

The Dynamic Prompt Generator receives the ContextBuilder output and
produces a prompt specifically for the current scene.

The process is:

``` text
CV Outputs
   +
Scene Knowledge
   ↓
Structured Context
   ↓
DynamicPromptGenerator
   ↓
Natural-Language Prompt
```

The prompt includes scene context, expected activities, contextual
concerns, observed objects, tracking summaries, and explicit decision
rules.

Raw tracking coordinates and repetitive track-level metadata are
intentionally omitted from the Qwen prompt in favor of compact semantic
summaries.

------------------------------------------------------------------------

## Qwen2.5-VL

The Qwen model used by this project is:

``` text
Qwen/Qwen2.5-VL-3B-Instruct
```

The project uses the Hugging Face Transformers interface and
`qwen-vl-utils`.

The Qwen wrapper accepts:

``` python
qwen.generate(
    frames=qwen_frames,
    prompt=dynamic_prompt,
)
```

The current multimodal configuration uses **four representative
frames**.

------------------------------------------------------------------------

# Kaggle Setup

Because Qwen inference is GPU-intensive, the recommended reproducible
environment for the current research pipeline is a Kaggle Notebook with
GPU acceleration.

The basic setup is:

``` text
GitHub repository
       ↓
Kaggle Notebook
       ↓
GPU enabled
       ↓
Clone repository
       ↓
Add ShanghaiTech dataset
       ↓
Install dependencies
       ↓
Run pipeline
       ↓
Generate dynamic prompt
       ↓
Run Qwen inference
```

## 1. Create a Kaggle Notebook

Create a new Kaggle Notebook.

Enable:

``` text
Notebook Settings
→ Accelerator
→ GPU
```

Verify the GPU:

``` python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

The expected CUDA result is:

``` text
CUDA available: True
```

------------------------------------------------------------------------

## 2. Clone the GitHub Repository

In a Kaggle cell:

``` python
!git clone https://github.com/<YOUR_USERNAME>/Context-Aware-Anomaly-Detection.git
```

Then:

``` python
%cd /kaggle/working/Context-Aware-Anomaly-Detection
```

Replace `<YOUR_USERNAME>` with the GitHub account containing the
repository.

------------------------------------------------------------------------

## 3. Install Dependencies

Run:

``` python
!pip install -q transformers qwen-vl-utils ultralytics supervision pyyaml pillow
```

Kaggle normally provides PyTorch, but CUDA should still be verified
using the previous cell.

------------------------------------------------------------------------

## 4. Make the Repository Importable

``` python
import sys

sys.path.append(
    "/kaggle/working/Context-Aware-Anomaly-Detection"
)
```

------------------------------------------------------------------------

## 5. Add the Dataset

In Kaggle:

``` text
Add Input
```

Add the ShanghaiTech dataset.

Then inspect the Kaggle input directory:

``` python
from pathlib import Path

for path in Path("/kaggle/input").iterdir():
    print(path)
```

Find the directory containing:

``` text
frames/
label/
SHANGHAI_train.txt
```

Set it as:

``` python
dataset_path = Path(
    "/kaggle/input/<DATASET-NAME>/SHANGHAI_TRAIN"
)

print("Dataset:", dataset_path)
print("Frames:", (dataset_path / "frames").exists())
print("Labels:", (dataset_path / "label").exists())
```

The exact `/kaggle/input/...` path depends on the dataset name used when
adding the input.

------------------------------------------------------------------------

# Running the Pipeline on Kaggle

## Step 1 --- Select a sequence

For example:

``` python
from pathlib import Path

sequence_id = "11_0176"

frames_dir = (
    dataset_path
    / "frames"
    / sequence_id
)

frame_paths = sorted(
    frames_dir.glob("*.jpg")
)

print("Sequence:", sequence_id)
print("Number of frames:", len(frame_paths))
```

Other sequence IDs can be used in the same way.

------------------------------------------------------------------------

## Step 2 --- Select four representative frames

The current Qwen configuration uses four representative frames:

``` python
from PIL import Image

selected_indices = [
    0,
    len(frame_paths) // 3,
    (2 * len(frame_paths)) // 3,
    len(frame_paths) - 1,
]

selected_frame_paths = [
    frame_paths[i]
    for i in selected_indices
]

qwen_frames = [
    Image.open(path).convert("RGB")
    for path in selected_frame_paths
]

print("Selected frames:")

for path in selected_frame_paths:
    print(path)

print("Number of Qwen frames:", len(qwen_frames))
```

The final line should report:

``` text
Number of Qwen frames: 4
```

------------------------------------------------------------------------

## Step 3 --- Run the CV pipeline

The existing repository components should be run in this order:

``` text
CLIP
 ↓
YOLO11n
 ↓
ByteTrack
 ↓
ContextBuilder
```

The outputs required by ContextBuilder are:

``` python
scene_id
scene_similarity
scene_info
tracking_data
```

Then:

``` python
from src.context.context_builder import ContextBuilder

context_builder = ContextBuilder()

context = context_builder.build_context(
    scene_id=scene_id,
    scene_similarity=scene_similarity,
    scene_info=scene_info,
    tracking_data=tracking_data,
)
```

Inspect the result:

``` python
from pprint import pprint

pprint(context)
```

The resulting dictionary should follow ContextBuilder schema `1.0`.

------------------------------------------------------------------------

# Dynamic Prompt + Qwen Inference Code

The following is the core Kaggle workflow used for dynamic prompt
generation and Qwen inference.

## Generate the Dynamic Prompt

``` python
from src.prompting.dy_prompt_generator import DynamicPromptGenerator

prompt_generator = DynamicPromptGenerator()

test_prompt = prompt_generator.generate_prompt(context)

print("
Generated Dynamic Prompt:")
print(test_prompt)

# Ensure the newly generated prompt is the prompt sent to Qwen.
dynamic_prompt = test_prompt
```

The explicit assignment:

``` python
dynamic_prompt = test_prompt
```

is useful in a notebook because variables remain in memory between
cells.

------------------------------------------------------------------------

## Prepare the Four Frames

``` python
from PIL import Image

qwen_frames = [
    Image.open(path).convert("RGB")
    for path in selected_frame_paths
]

print("Selected frames:")

for path in selected_frame_paths:
    print(path)

print("Number of frames:", len(qwen_frames))
```

------------------------------------------------------------------------

## Initialize Qwen

``` python
from src.qwen.qwen_model import QwenModel

qwen = QwenModel()
```

The wrapper loads:

``` text
Qwen/Qwen2.5-VL-3B-Instruct
```

and uses CUDA when available.

------------------------------------------------------------------------

## Run Qwen Inference

``` python
print("
Sending sequence to Qwen...")

qwen_response = qwen.generate(
    frames=qwen_frames,
    prompt=dynamic_prompt,
)

print("
Qwen Response:")
print(qwen_response)
```

------------------------------------------------------------------------

## Complete Core Inference Cell

For convenience, the dynamic-prompt and Qwen portion can be run as one
cell:

``` python
from PIL import Image

from src.prompting.dy_prompt_generator import DynamicPromptGenerator
from src.qwen.qwen_model import QwenModel

# ---------------------------------------------------------
# Prepare four representative frames
# ---------------------------------------------------------

qwen_frames = [
    Image.open(path).convert("RGB")
    for path in selected_frame_paths
]

print("Selected frames:")
for path in selected_frame_paths:
    print(path)

print("Number of frames:", len(qwen_frames))

# ---------------------------------------------------------
# Generate dynamic reasoning prompt
# ---------------------------------------------------------

prompt_generator = DynamicPromptGenerator()

test_prompt = prompt_generator.generate_prompt(context)

print("
Generated Dynamic Prompt:")
print(test_prompt)

# Make sure Qwen receives the newly generated prompt.
dynamic_prompt = test_prompt

# ---------------------------------------------------------
# Qwen inference
# ---------------------------------------------------------

qwen = QwenModel()

print("
Sending sequence to Qwen...")

qwen_response = qwen.generate(
    frames=qwen_frames,
    prompt=dynamic_prompt,
)

print("
Qwen Response:")
print(qwen_response)
```

------------------------------------------------------------------------

# Reproducibility Notes

### GPU

Qwen inference should be run with a CUDA-capable GPU for practical
execution.

### First model run

The first Qwen initialization downloads the model from Hugging Face.
YOLO11n weights are also downloaded if they are not already cached.

### Dataset paths

Kaggle dataset paths are environment-specific. The repository path is
normally:

``` text
/kaggle/working/Context-Aware-Anomaly-Detection
```

while attached datasets appear under:

``` text
/kaggle/input/
```

Only the dataset path normally needs to be changed when another
researcher creates a new Kaggle notebook.

### Sequence switching

Kaggle notebook variables persist between cells. When switching
sequences, rebuild all sequence-dependent variables:

``` python
frames_dir
frame_paths
selected_indices
selected_frame_paths
qwen_frames
```

For example:

``` python
sequence_id = "06_0155"

frames_dir = dataset_path / "frames" / sequence_id

frame_paths = sorted(
    frames_dir.glob("*.jpg")
)

selected_indices = [
    0,
    len(frame_paths) // 3,
    (2 * len(frame_paths)) // 3,
    len(frame_paths) - 1,
]

selected_frame_paths = [
    frame_paths[i]
    for i in selected_indices
]

qwen_frames = [
    Image.open(path).convert("RGB")
    for path in selected_frame_paths
]
```

Always verify:

``` python
print(frames_dir)

for path in selected_frame_paths:
    print(path)

print("Frame count:", len(qwen_frames))
```

This ensures that all inputs correspond to the currently selected
sequence.

------------------------------------------------------------------------

# Expected Output

A successful Qwen run returns a natural-language response containing a
classification and explanation.

Typical structure:

``` text
Classification: NORMAL

Observed Activity:
...

Reasoning:
...
```

or:

``` text
Classification: ANOMALOUS

Observed Activity:
...

Reasoning:
...
```

or:

``` text
Classification: UNCERTAIN

Observed Activity:
...

Reasoning:
...
```

The exact wording is generated by Qwen.

------------------------------------------------------------------------

# Current Scope

The current implementation focuses on:

-   ShanghaiTech surveillance sequences
-   13-scene scene recognition
-   pretrained CLIP scene recognition
-   YOLO11n object detection
-   ByteTrack multi-object tracking
-   structured scene/domain knowledge
-   ContextBuilder schema v1.0
-   automatic dynamic prompt generation
-   four-frame Qwen2.5-VL inference
-   natural-language classification and reasoning
-   Kaggle-based GPU execution

The current system is intended primarily for research experimentation
and demonstration.

------------------------------------------------------------------------

# Future Work

Potential extensions include:

-   Gradio-based video upload interface
-   interactive web demonstration
-   FastAPI inference backend
-   remote GPU deployment
-   real-time CCTV input
-   continuous video inference
-   multi-camera tracking
-   richer temporal activity representations
-   automated event logging
-   alert generation
-   dashboard visualization
-   additional surveillance domains
-   evaluation on additional datasets
-   quantitative anomaly-detection metrics

------------------------------------------------------------------------

# Acknowledgements

This project makes use of established open-source technologies
including:

-   CLIP
-   Ultralytics YOLO
-   ByteTrack
-   PyTorch
-   Hugging Face Transformers
-   Qwen2.5-VL
-   Pillow
-   NumPy

The project also uses the ShanghaiTech Campus dataset for research
experimentation.

Please follow the respective licenses, citation requirements, and usage
terms of the underlying models, libraries, and dataset.

------------------------------------------------------------------------


