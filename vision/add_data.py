import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os 

os.environ["FIFTYONE_HOME"] = "D:/PCVision/Nhom7-NguyenTungLam/vision/fiftyone"

# ── 1. Load a detection dataset from the zoo ──────────────────────────────────
# COCO-2017 is the standard benchmark for YOLO training.
# Use "train" split for training data; swap to "validation" for val split.
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],   # bounding boxes only – what YOLO needs
    max_samples=1000,               # increase / remove cap for a full run
    shuffle=True,
    dataset_name="yolov10-train",
)

# ── 2. (Optional) Filter to specific classes ──────────────────────────────────
CLASSES = ["person", "car", "dog", "tree", "banner"]   # swap for any COCO class names you need

view = dataset.filter_labels(
    "ground_truth",
    F("label").is_in(CLASSES),
).match(
    F("ground_truth.detections").length() > 0   # drop images with no kept labels
)

# ── 3. Export in YOLOv5/v8/v10 format (same directory layout) ─────────────────
EXPORT_DIR = "./yolov10_dataset_local" # Changed to a local directory

view.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,   # YOLOv10 uses identical folder layout
    label_field="ground_truth",
    classes=CLASSES,
    split="train",
)

# ── 4. Repeat for the validation split ────────────────────────────────────────
val_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    max_samples=100,
    dataset_name="yolov10-val",
)

val_view = val_dataset.filter_labels(
    "ground_truth",
    F("label").is_in(CLASSES),
).match(
    F("ground_truth.detections").length() > 0
)

val_view.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=CLASSES,
    split="val",
)

print(f"Dataset exported to: {EXPORT_DIR}")
print("Expected layout:")
print("  yolov10_dataset/")
print("  ├── images/train/  & images/val/")
print("  ├── labels/train/  & labels/val/")
print("  └── dataset.yaml")

# ── 5. (Optional) Launch FiftyOne App to verify ───────────────────────────────
session = fo.launch_app(view)