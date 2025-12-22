from pathlib import Path
import json
import cv2
from PIL import Image
from pix2text import Pix2Text

# -------------------------
# Config
# -------------------------
PAGE_IMG = "test_data/03.png"
OUT_DIR = Path("output_pix2text_detect_page03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD = 2  # padding ตอน crop

# -------------------------
# Helpers
# -------------------------
def bgr_to_pil(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def crop_bgr(img_bgr, xyxy, pad=0):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w, h)
    return img_bgr[y1:y2, x1:x2], [x1, y1, x2, y2]

def try_get_bbox(item):
    """
    พยายามดึง bbox จาก item ของ Pix2Text หลายรูปแบบ:
    - item['bbox'] อาจเป็น [x1,y1,x2,y2] หรือ [[x,y],...]
    - item['position'] / item['box'] / item['polygon']
    คืนค่า: (xyxy_int หรือ None)
    """
    if not isinstance(item, dict):
        return None

    for key in ("bbox", "box", "position", "polygon"):
        if key not in item:
            continue
        b = item[key]
        if b is None:
            continue

        # case 1: [x1,y1,x2,y2]
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x1, y1, x2, y2 = b
            return [x1, y1, x2, y2]

        # case 2: [[x,y], [x,y], [x,y], [x,y]] polygon
        if isinstance(b, (list, tuple)) and len(b) >= 4 and isinstance(b[0], (list, tuple)) and len(b[0]) == 2:
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            return [min(xs), min(ys), max(xs), max(ys)]

    return None

def get_type_and_text(item):
    """
    ดึง type + content จาก item
    """
    if not isinstance(item, dict):
        return "unknown", str(item)

    t = item.get("type") or item.get("cls") or item.get("category") or "unknown"

    # content อาจอยู่หลาย key
    latex = item.get("latex")
    text = item.get("text")
    if latex:
        return "formula", latex
    if text:
        return "text", text

    # fallback
    return t, json.dumps(item, ensure_ascii=False)

# -------------------------
# Main
# -------------------------
def main():
    page_bgr = cv2.imread(PAGE_IMG)
    if page_bgr is None:
        raise ValueError(f"Cannot read image: {PAGE_IMG}")

    p2t = Pix2Text.from_config()  # init once

    pil = bgr_to_pil(page_bgr)

    # Pix2Text detect+recognize ทั้งหน้า
    res = p2t.recognize(pil)

    # เซฟ raw output ก่อน (สำคัญมาก)
    (OUT_DIR / "pix2text_raw.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # ถ้า res เป็น string/dict -> เรา crop ไม่ได้แน่ ๆ แต่ยังดูผลได้
    if not isinstance(res, list):
        (OUT_DIR / "pix2text_text.txt").write_text(str(res), encoding="utf-8")
        print("[WARN] Pix2Text output is not a list. Saved raw + text only.")
        return

    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    items_out = []
    crop_count = 0

    # sort ถ้ามี bbox
    sortable = []
    for i, item in enumerate(res):
        bbox = try_get_bbox(item)
        sortable.append((i, item, bbox))
    sortable.sort(key=lambda it: (it[2][1], it[2][0]) if it[2] else (10**9, 10**9))

    for new_idx, (i, item, bbox) in enumerate(sortable, start=1):
        typ, content = get_type_and_text(item)

        entry = {
            "index": new_idx,
            "type": typ,
            "content": content,
            "bbox_xyxy": bbox,
            "raw_item": item,
        }

        # ถ้ามี bbox -> crop ภาพออกมา
        if bbox is not None:
            crop, bbox_int = crop_bgr(page_bgr, bbox, pad=PAD)
            crop_count += 1
            # ตั้งชื่อให้เห็นว่าเป็น text หรือ formula
            fname = f"{crop_count:03d}_{typ}.jpg"
            crop_path = crops_dir / fname
            cv2.imwrite(str(crop_path), crop)

            # เขียนผลลัพธ์ text ของ crop แยกไฟล์ด้วย
            (crops_dir / f"{crop_count:03d}_{typ}.txt").write_text(content, encoding="utf-8")

            entry["bbox_xyxy"] = bbox_int
            entry["crop_image"] = str(crop_path.relative_to(OUT_DIR))

        items_out.append(entry)

    # summary
    summary = {
        "page_image": PAGE_IMG,
        "items_total": len(res),
        "crops_saved": crop_count,
        "items": items_out
    }

    (OUT_DIR / "pix2text_items.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[OK] items={len(res)} crops_saved={crop_count}")
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
