from ultralytics import YOLO
from pix2text import Pix2Text
from PIL import Image
import os
import uuid
from datetime import datetime
import json

# (optional) ถ้ามี opencv จะเซฟภาพจาก results.plot() ได้ง่ายขึ้น
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_yolo_annotated_image(result, save_path: str):
    """
    result: results[0] ของ ultralytics
    save_path: path ที่จะบันทึกภาพ (png/jpg)
    """
    im = result.plot()  # numpy array (BGR)
    if HAS_CV2:
        cv2.imwrite(save_path, im)
    else:
        # fallback: แปลง BGR -> RGB แล้วเซฟด้วย PIL
        rgb = im[:, :, ::-1]
        Image.fromarray(rgb).save(save_path)


# --- YOLO ---
model = YOLO("best.pt")
name_to_id = {v: k for k, v in model.names.items()}

keep = [
    name_to_id["qt"], name_to_id["pb"], name_to_id["img"],
    name_to_id["c1"], name_to_id["c2"], name_to_id["c3"], name_to_id["c4"]
]

img_path = "test_data/12.png"
results = model.predict(
    img_path,
    classes=keep,
    conf=0.35,
    iou=0.45,
    agnostic_nms=False,
    augment=False,
)

r = results[0]

# --- สร้างโฟลเดอร์ผลลัพธ์ตามวันเวลา ---
ts = datetime.now().strftime("%d%m%Y%H%M%S")  # เช่น 20122025091530
base_dir = os.path.join("result", ts)
yolo_dir = base_dir
crop_dir = os.path.join(base_dir, "qt_crops")
p2t_dir = os.path.join(base_dir, "p2t_out")

ensure_dir(yolo_dir)
ensure_dir(crop_dir)
ensure_dir(p2t_dir)

# --- เซฟภาพ YOLO ที่ตีกรอบแล้ว ---
yolo_out_path = os.path.join(yolo_dir, "yolo_detect.png")
save_yolo_annotated_image(r, yolo_out_path)
print("Saved YOLO detect image:", yolo_out_path)

# --- Pix2Text ---
# แนะนำให้บังคับ CPU เพื่อกัน error CUDAExecutionProvider
p2t = Pix2Text.from_config(device="cpu")

img = Image.open(img_path).convert("RGB")

qt_id = name_to_id["qt"]
qt_mask = (r.boxes.cls == qt_id)

qt_boxes = r.boxes.xyxy[qt_mask].cpu().numpy()

# เรียง qt จากบนลงล่าง (กันสลับข้อ)
qt_boxes = sorted(qt_boxes, key=lambda b: (b[1], b[0]))

for idx, (x1, y1, x2, y2) in enumerate(qt_boxes):
    pad = 6
    x1i = max(0, int(x1) - pad)
    y1i = max(0, int(y1) - pad)
    x2i = min(img.width,  int(x2) + pad)
    y2i = min(img.height, int(y2) + pad)

    crop = img.crop((x1i, y1i, x2i, y2i))

    # --- เซฟรูป crop (ที่ส่งเข้า Pix2Text) ---
    crop_path = os.path.join(crop_dir, f"qt_{idx:02d}.png")
    crop.save(crop_path)

    # --- ส่งเข้า Pix2Text ---
    outs = p2t.recognize_text_formula(crop_path, resized_shape=768, return_text=True)

    # --- เซฟผลลัพธ์ Pix2Text ---
    # บางเครื่องคืนเป็น str, บางเครื่องคืนเป็น list/dict -> เซฟทั้งแบบ txt + json ให้เลย
    txt_path = os.path.join(p2t_dir, f"qt_{idx:02d}.txt")
    json_path = os.path.join(p2t_dir, f"qt_{idx:02d}.json")

    # TXT (อ่านง่าย)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"QT #{idx}\n")
        f.write(f"crop: {crop_path}\n\n")
        if isinstance(outs, str):
            f.write(outs)
        else:
            f.write(json.dumps(outs, ensure_ascii=False, indent=2))

    # JSON (เก็บโครงสร้าง)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)

    print(f"Saved QT #{idx}: crop={crop_path}  p2t_txt={txt_path}  p2t_json={json_path}")

print("\nAll results saved under:", base_dir)
