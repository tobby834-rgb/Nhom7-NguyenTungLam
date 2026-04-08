# ── 6. Train và tự động lưu model ─────────────────────────────────────────────
from ultralytics import YOLO
import shutil
import os

# Load model và train
model = YOLO(r"D:\PCVision\Nhom7-NguyenTungLam\runs\detect\yolov10_runs\my_model\weights\last.pt")

results = model.train(
    data="./yolov10_dataset_local/dataset.yaml",
    epochs=100,
    imgsz=640,
    project="runs/detect/yolov10_runs",  # 👈 sửa lại
    name="my_model",
    resume=True
)

# ── 7. Lấy đường dẫn model tốt nhất ──────────────────────────────────────────
best_model_path = results.save_dir / "weights" / "best.pt"
last_model_path = results.save_dir / "weights" / "last.pt"

print(f"✅ Best model: {best_model_path}")
print(f"✅ Last model: {last_model_path}")

# ── 8. Copy ra thư mục riêng để dễ tải ───────────────────────────────────────
OUTPUT_DIR = "./exported_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

shutil.copy(best_model_path, f"{OUTPUT_DIR}/yolov10_best.pt")
shutil.copy(last_model_path, f"{OUTPUT_DIR}/yolov10_last.pt")

print(f"📦 Model đã được lưu tại: {OUTPUT_DIR}/")