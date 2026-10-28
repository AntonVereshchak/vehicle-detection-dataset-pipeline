# Vehicle Detection Dataset: Preparation, Annotation Pipeline & Quality Assessment

## Project scope

The original task was to train a vehicle detector for two distance ranges: **0–200 m** and **200–400 m**. My work focused on preparing and reviewing the training data and building a practical annotation workflow.

This README covers the dataset preparation and annotation work completed so far. It does **not** present a finished detector or distance-based evaluation results.

## 1. Data review

The source dataset contains **five videos**: four for training and one reserved for model evaluation.

After reviewing the videos, I found several different road scenarios, with changes in traffic density, camera movement, object scale, and partial occlusions.

I used a Python script to collect video metadata (FPS, resolution, duration, and total frame count). I reviewed camera movement, camera angle, and object density manually.

| Video | Role | FPS | Resolution | Duration (s) | Frames | Camera movement | Camera angle | Object density |
|---|---|---:|---|---:|---:|---|---|---|
| `13722965_2160_3840_30fps.mp4` | Evaluation | 30.00 | 2160×3840 | 30.60 | 918 | Forward | Top-down | Low |
| `3405804-uhd_3840_2160_30fps.mp4` | Train D | 29.97 | 3840×2160 | 24.49 | 734 | Backward | Top-down | Medium |
| `5382494-uhd_3840_2160_24fps.mp4` | Train B | 24.00 | 3840×2160 | 31.25 | 750 | Forward | Near-horizontal | Low |
| `8457857-uhd_3840_2160_24fps.mp4` | Train C | 23.98 | 3840×2160 | 16.89 | 405 | Tilt up (stationary) | Near-nadir to horizontal | High |
| `8968356-hd_1920_1080_30fps.mp4` | Train A | 29.97 | 1920×1080 | 19.22 | 576 | Static | Near top-down | Medium |

Based on the different scenes and viewing conditions, I considered these videos suitable for the task.

## 2. Data preparation and frame selection

The goal was to build a representative set of frames without keeping too many near-duplicates.

I created two Python scripts: one for extracting frames and another for manually selecting the frames to keep.

My selection criteria were:

- Spread selected frames across the full length of each video.
- Keep different vehicle positions and traffic situations.
- Avoid nearly identical frames where possible.
- Keep challenging scenes, including partially occluded vehicles.
- Include vehicles at different image scales.

| Video | Selected frames |
|---|---:|
| Train A | 70 |
| Train B | 20 |
| Train C | 44 |
| Train D | 44 |
| **Total** | **178** |

These 178 frames were selected for the auto-annotation workflow. **Only manually reviewed frames were included in the verified dataset** described below.

## 3. Model-assisted auto-annotation

I used **YOLO11m** with pretrained COCO weights to generate initial bounding boxes. I added **SAHI sliced inference** because many vehicles in aerial footage appear very small.

The COCO classes `car`, `motorcycle`, `bus`, and `truck` were mapped to one annotation class: **`Vehicle`**.

| Setting | Value |
|---|---|
| Detection model | YOLO11m |
| Inference method | SAHI sliced inference |
| Slice size | 640 × 640 px |
| Slice overlap | 20% |
| Confidence threshold | 0.10 |
| Output class | `Vehicle` |

The model was used to **speed up annotation**, not as a final detector meeting the model-size requirement of the original task.

## 4. Manual review and annotation guidelines

I imported the auto-generated labels into **CVAT**, reviewed the frames, and corrected the annotations manually.

I followed these rules:

- Use a single class: `Vehicle`.
- Draw bounding boxes around the visible part of each vehicle.
- Annotate partially occluded vehicles and vehicles cut off by the image border when they can still be identified confidently.
- Add vehicles missed by the model.
- Remove incorrect detections.
- Adjust inaccurate bounding boxes.
- Do not add uncertain labels; flag ambiguous cases for another review where possible.
- Apply the same rules across all videos, regardless of scale, camera angle, or traffic density.

### Pilot: 10 manually reviewed frames

Before processing the larger set, I reviewed **10 frames from Train A** to estimate the annotation workload and understand the main errors in the auto-generated labels.

| Pilot setting | Value |
|---|---|
| Dataset | Train A — highway interchange |
| Frames | 10 |
| Auto-annotation | YOLO11m + SAHI |
| Slice size | 640 × 640 px |
| Overlap | 20% |
| Confidence threshold | 0.10 |
| Review tool | CVAT |
| Matching threshold | IoU ≥ 0.5 |

**Pilot results** (automatic proposals compared with manually reviewed labels):

| Metric | Result |
|---|---:|
| Auto-generated bounding boxes | 137 |
| Objects after manual review | 386 |
| True positives (TP) | 135 |
| False positives (FP) | 2 |
| False negatives (FN) | 251 |
| **Precision** | **98.54%** |
| **Recall** | **34.97%** |
| **F1-score** | **51.63%** |

**Time spent in CVAT:**

| Measure | Result |
|---|---:|
| Frames reviewed | 10 |
| Total review time | ~12 minutes |
| Average per frame | ~72 seconds |
| Final annotations | 386 |
| Average objects per frame | 38.6 |

The auto-annotation model found roughly **35%** of the vehicles in the reviewed labels. Most of its proposed boxes were correct, but it missed many objects. So the predictions were useful as a starting point, **not** as a replacement for a full manual review of each frame.

This also affects workload planning. Frame count alone is not enough: I would also consider **object density, scene complexity, and the quality of the initial auto-labels**.

The pilot metrics are based on my manually reviewed annotations, not an independently audited ground-truth set.

### Verified dataset

The annotation workflow used was:

**Auto-annotation → CVAT import → Manual review → Corrections → Export → Verified dataset**

| Video | Manually reviewed frames | Annotated objects |
|---|---:|---:|
| Train A | 35 | 1,492 |
| Train B | 20 | 201 |
| Train C | 44 | 1,143 |
| Train D | 5 | 167 |
| **Total** | **104** | **3,003** |

The remaining auto-labeled frames were **not** included in this verified set.

## 5. Suggested workflow for future iterations

For a larger project, I would use an iterative process:

**Verified dataset → Model training → Evaluation and error analysis → Improved auto-annotation → Manual review and correction → QC → Updated verified dataset**

This is a **proposed next-stage workflow**, not a claim that all these steps were completed during this task. I carried out manual review and some repeat checking, but **not an independent QC audit**.

## 6. Quality control plan for production

For a production annotation workflow, I would suggest:

1. First-pass review of each selected frame by an annotator.
2. Sample-based second-pass QC.
3. Tracking errors by type.
4. Returning problematic frames for correction.
5. Reviewing recurring errors and their causes.
6. Updating annotation guidelines when needed.

I would track these quality checks separately:

- Missed vehicles.
- Incorrect detections.
- Bounding-box accuracy.
- Frames with critical errors that need another review.

## 7. Current limitations

This submission focuses on **data preparation, annotation, and process design**. The following parts of the original detector task are **not completed here**:

- A final trained detector with fewer than 500,000 parameters.
- Final evaluation on the reserved evaluation video.
- Detection metrics for the 0–200 m and 200–400 m distance ranges.
- A final model demo video.

The verified dataset has 104 frames from four videos, with uneven coverage across the videos. Frames from the same video may also be similar. The pilot quality figures are useful for planning the annotation workflow but should not be treated as independent model-performance results.
