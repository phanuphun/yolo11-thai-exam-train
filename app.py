from ultralytics import YOLO
from pathlib import Path
import cv2
import json
from PIL import Image

from pix2text import Pix2Text

# -------------------------
# Config
# -------------------------
MODEL_PATH = "best.pt"
PAGE_IMG = "test_data/01.png"

# output/<PAGE_STEM>/problemX/...
OUT_ROOT = Path("output")
PAGE_STEM = Path(PAGE_IMG).stem  # "01"
OUT_PAGE = OUT_ROOT / PAGE_STEM
OUT_PAGE.mkdir(parents=True, exist_ok=True)

# Stage1 (pb)
CONF_PB = 0.10
IOU_PB = 0.45

# Stage2 incremental search schedule
CONF_STAGE2_START = 0.50
CONF_STAGE2_MIN = 0.10
CONF_STAGE2_STEP = 0.10

# NMS tighten
IOU_STAGE2_NMS = 0.35
MAX_DET = 300

# Post-process dedupe in same class
DEDUP_IOU = 0.85

CLS_PROBLEM = "pb"
CLS_QUESTION = "qt"
CLS_CHOICES = ["c1", "c2", "c3", "c4"]
CLS_IMG = "img"     # optional
CLS_LATEX = "lt"    # latex region

# ✅ default style (Ultralytics plot)
LINE_WIDTH = 1
FONT_SIZE = 2

USE_PIX2TEXT = True
PIX2TEXT_READ_FULL_QC = False


# -------------------------
# Utils
# -------------------------
def add_label_suffix(path: Path) -> Path:
    return path.with_name(f"{path.stem}_label{path.suffix}")

def fname(*parts: str, ext=".jpg") -> str:
    """
    สร้างชื่อไฟล์แบบมี prefix PAGE_STEM เสมอ
    เช่น fname("page_pb_label") -> "01_page_pb_label.jpg"
    """
    safe = "_".join([p for p in parts if p]).replace(" ", "_")
    return f"{PAGE_STEM}_{safe}{ext}"

def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def crop_bgr(img_bgr, xyxy, pad=0):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w, h)
    return img_bgr[y1:y2, x1:x2], [x1, y1, x2, y2]

def sort_boxes_top_left(box_list):
    return sorted(box_list, key=lambda b: (b["xyxy"][1], b["xyxy"][0]))

def ensure_class(name_to_id, name: str):
    if name not in name_to_id:
        raise ValueError(f'Class "{name}" not found in model.names. Available: {list(name_to_id.keys())}')
    return name_to_id[name]

def yolo_detect(model, img_path_or_bgr, class_ids, conf=0.35, iou=0.45, max_det=300):
    """
    return: (dets, result_obj)
      dets = list of {cls_id, cls_name, conf, xyxy}
    """
    results = model.predict(
        img_path_or_bgr,
        classes=class_ids,
        conf=conf,
        iou=iou,
        max_det=max_det,
        agnostic_nms=False,
        augment=False,
        verbose=False,
    )
    r = results[0]
    boxes = r.boxes
    out = []
    if boxes is None or len(boxes) == 0:
        return out, r

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf_i = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        out.append({
            "cls_id": cls_id,
            "cls_name": model.names[cls_id],
            "conf": conf_i,
            "xyxy": [x1, y1, x2, y2],
        })
    return out, r


# -------------------------
# Dedupe (same class)
# -------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def dedupe_dets_same_class(dets, iou_thr=0.85):
    kept = []
    dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)

    for d in dets_sorted:
        dup = False
        for k in kept:
            if d["cls_name"] == k["cls_name"]:
                if iou_xyxy(d["xyxy"], k["xyxy"]) >= iou_thr:
                    dup = True
                    break
        if not dup:
            kept.append(d)
    return kept


# -------------------------
# Retry history save (✅ YOLO default plot)
# -------------------------
def save_attempt(history_dir: Path, attempt_idx: int, tag: str, result_obj, dets_after_dedupe: list[dict], meta: dict, problem_idx: int):
    """
    สร้างไฟล์คู่:
      - <PAGE>_problemX_Axx__{tag}.jpg   (YOLO default)
      - <PAGE>_problemX_Axx__{tag}.json
    """
    history_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{PAGE_STEM}_problem{problem_idx}_A{attempt_idx:02d}__{tag}"
    img_path = history_dir / f"{stem}.jpg"
    json_path = history_dir / f"{stem}.json"

    overlay = result_obj.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)
    cv2.imwrite(str(img_path), overlay)

    payload = {
        "attempt_index": attempt_idx,
        "tag": tag,
        "meta": meta,
        "detections_deduped": dets_after_dedupe,
        "counts_deduped": {
            "total": len(dets_after_dedupe),
            "by_class": {
                CLS_QUESTION: sum(1 for d in dets_after_dedupe if d["cls_name"] == CLS_QUESTION),
                "c1": sum(1 for d in dets_after_dedupe if d["cls_name"] == "c1"),
                "c2": sum(1 for d in dets_after_dedupe if d["cls_name"] == "c2"),
                "c3": sum(1 for d in dets_after_dedupe if d["cls_name"] == "c3"),
                "c4": sum(1 for d in dets_after_dedupe if d["cls_name"] == "c4"),
                CLS_IMG: sum(1 for d in dets_after_dedupe if d["cls_name"] == CLS_IMG),
            }
        }
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return img_path, json_path


# -------------------------
# Pix2Text helpers
# -------------------------
def bgr_to_pil(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def parse_pix2text_output(res):
    latex_list = []
    text_parts = []

    if isinstance(res, str):
        return {"text": res, "latex_list": [], "raw": res}

    if isinstance(res, dict):
        t = res.get("text") or res.get("result") or res.get("final") or ""
        return {"text": str(t), "latex_list": [], "raw": res}

    if isinstance(res, list):
        for it in res:
            if not isinstance(it, dict):
                continue
            if "latex" in it and it["latex"]:
                latex_list.append(it["latex"])
            if "text" in it and it["text"]:
                if it.get("type") in ("formula", "math", "latex"):
                    latex_list.append(it["text"])
                else:
                    text_parts.append(it["text"])
        return {"text": " ".join(text_parts).strip(), "latex_list": latex_list, "raw": res}

    return {"text": str(res), "latex_list": [], "raw": res}

def pix2text_read_image(p2t: Pix2Text, pil_img: Image.Image):
    try:
        res = p2t.recognize(pil_img)
    except Exception as e:
        return {"text": "", "latex_list": [], "raw": f"Pix2Text error: {repr(e)}"}
    return parse_pix2text_output(res)


# -------------------------
# Incomplete rename
# -------------------------
def rename_dir_incomplete(problem_dir: Path, missing: list[str]):
    suffix = "__INCOMPLETE_" + "_".join(missing)[:120]
    new_dir = Path(str(problem_dir) + suffix)
    if new_dir.exists():
        i = 2
        while Path(str(new_dir) + f"_{i}").exists():
            i += 1
        new_dir = Path(str(new_dir) + f"_{i}")
    problem_dir.rename(new_dir)
    return new_dir


# -------------------------
# Incremental Stage2 (with history)
# -------------------------
def conf_schedule_list(start=0.5, step=0.1, minv=0.1):
    vals = []
    v = start
    while v >= minv - 1e-9:
        vals.append(round(v, 2))
        v -= step
    return vals

def pick_one_det(dets, prefer="top_left"):
    if not dets:
        return None
    if prefer == "best_conf":
        return sorted(dets, key=lambda d: d["conf"], reverse=True)[0]
    return sorted(dets, key=lambda d: (d["xyxy"][1], d["xyxy"][0]))[0]

def detect_incremental_stage2(
    *,
    model,
    pb_crop,
    qt_id,
    choice_ids_map,
    img_id,
    history_dir: Path,
    problem_idx: int,
):
    confs = conf_schedule_list(CONF_STAGE2_START, CONF_STAGE2_STEP, CONF_STAGE2_MIN)

    found_qt = None
    found_choices = {c: None for c in choice_ids_map.keys()}
    found_imgs = []
    used_conf = {}
    last_r = None

    history_records = []
    attempt_idx = 0

    # ---- Pass 0: all classes at 0.50 ----
    all_ids = [qt_id] + list(choice_ids_map.values()) + ([img_id] if img_id is not None else [])
    dets, r = yolo_detect(model, pb_crop, all_ids, conf=confs[0], iou=IOU_STAGE2_NMS, max_det=MAX_DET)
    last_r = r
    dets_d = dedupe_dets_same_class(dets, iou_thr=DEDUP_IOU)

    imgp, jsonp = save_attempt(
        history_dir, attempt_idx,
        tag=f"all_conf{confs[0]:.2f}",
        result_obj=r,
        dets_after_dedupe=dets_d,
        meta={"mode": "all", "conf": confs[0], "iou": IOU_STAGE2_NMS, "max_det": MAX_DET, "dedup_iou": DEDUP_IOU},
        problem_idx=problem_idx
    )
    history_records.append({"attempt": attempt_idx, "img": str(imgp), "json": str(jsonp)})
    attempt_idx += 1

    qt_dets = [d for d in dets_d if d["cls_name"] == CLS_QUESTION]
    one_qt = pick_one_det(qt_dets, prefer="top_left")
    if one_qt:
        found_qt = one_qt
        used_conf["qt"] = confs[0]

    for c in choice_ids_map.keys():
        c_dets = [d for d in dets_d if d["cls_name"] == c]
        one = pick_one_det(c_dets, prefer="top_left")
        if one:
            found_choices[c] = one
            used_conf[c] = confs[0]

    if img_id is not None:
        found_imgs = [d for d in dets_d if d["cls_name"] == CLS_IMG]

    # ---- Step 1: หา qt ก่อนถ้ายังไม่เจอ ----
    if found_qt is None:
        for conf_try in confs[1:]:
            dets_qt, r_qt = yolo_detect(model, pb_crop, [qt_id], conf=conf_try, iou=IOU_STAGE2_NMS, max_det=MAX_DET)
            last_r = r_qt
            dets_qt_d = dedupe_dets_same_class(dets_qt, iou_thr=DEDUP_IOU)

            imgp, jsonp = save_attempt(
                history_dir, attempt_idx,
                tag=f"qt_conf{conf_try:.2f}",
                result_obj=r_qt,
                dets_after_dedupe=dets_qt_d,
                meta={"mode": "single", "target": "qt", "conf": conf_try, "iou": IOU_STAGE2_NMS, "max_det": MAX_DET, "dedup_iou": DEDUP_IOU},
                problem_idx=problem_idx
            )
            history_records.append({"attempt": attempt_idx, "img": str(imgp), "json": str(jsonp)})
            attempt_idx += 1

            qt_dets = [d for d in dets_qt_d if d["cls_name"] == CLS_QUESTION]
            one = pick_one_det(qt_dets, prefer="top_left")
            if one:
                found_qt = one
                used_conf["qt"] = conf_try
                break

    # ---- Step 2: หา choice ที่ขาดทีละตัว ----
    for c, cid in choice_ids_map.items():
        if found_choices[c] is not None:
            continue  # mark เจอแล้ว -> ไม่หาใหม่

        for conf_try in confs:
            dets_c, r_c = yolo_detect(model, pb_crop, [cid], conf=conf_try, iou=IOU_STAGE2_NMS, max_det=MAX_DET)
            last_r = r_c
            dets_c_d = dedupe_dets_same_class(dets_c, iou_thr=DEDUP_IOU)

            imgp, jsonp = save_attempt(
                history_dir, attempt_idx,
                tag=f"{c}_conf{conf_try:.2f}",
                result_obj=r_c,
                dets_after_dedupe=dets_c_d,
                meta={"mode": "single", "target": c, "conf": conf_try, "iou": IOU_STAGE2_NMS, "max_det": MAX_DET, "dedup_iou": DEDUP_IOU},
                problem_idx=problem_idx
            )
            history_records.append({"attempt": attempt_idx, "img": str(imgp), "json": str(jsonp)})
            attempt_idx += 1

            c_dets = [d for d in dets_c_d if d["cls_name"] == c]
            one = pick_one_det(c_dets, prefer="top_left")
            if one:
                found_choices[c] = one
                used_conf[c] = conf_try
                break

    # ---- Validate ----
    missing = []
    if found_qt is None:
        missing.append("no_qt")
    for c in choice_ids_map.keys():
        if found_choices[c] is None:
            missing.append(f"missing_{c}")

    return found_qt, found_choices, found_imgs, used_conf, last_r, missing, history_records


# -------------------------
# Main Flow
# -------------------------
def main():
    model = YOLO(MODEL_PATH)
    name_to_id = {v: k for k, v in model.names.items()}

    pb_id = ensure_class(name_to_id, CLS_PROBLEM)
    qt_id = ensure_class(name_to_id, CLS_QUESTION)
    lt_id = ensure_class(name_to_id, CLS_LATEX)
    img_id = name_to_id.get(CLS_IMG, None)

    choice_ids_map = {c: ensure_class(name_to_id, c) for c in CLS_CHOICES}

    p2t = Pix2Text.from_config() if USE_PIX2TEXT else None

    page = cv2.imread(PAGE_IMG)
    if page is None:
        raise ValueError(f"Cannot read image: {PAGE_IMG}")

    # Stage 1: Detect pb only
    pb_dets, pb_r = yolo_detect(model, PAGE_IMG, [pb_id], conf=CONF_PB, iou=IOU_PB, max_det=MAX_DET)
    if not pb_dets:
        print("No pb detected.")
        return

    pb_dets = sort_boxes_top_left(pb_dets)

    # page label (ชื่อไฟล์ตามหน้า)
    page_pb_label_path = OUT_PAGE / fname("page_pb_label")
    cv2.imwrite(str(page_pb_label_path), pb_r.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE))

    for idx, pb in enumerate(pb_dets, start=1):
        # output/<PAGE_STEM>/problemX/...
        problem_dir = OUT_PAGE / f"problem{idx}"
        q_dir = problem_dir / "question"
        c_dir = problem_dir / "choice"
        l_dir = problem_dir / "lartex"
        h_dir = problem_dir / "retry_history"

        q_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)
        l_dir.mkdir(parents=True, exist_ok=True)
        h_dir.mkdir(parents=True, exist_ok=True)

        # crop pb
        pb_crop, pb_xyxy_int = crop_bgr(page, pb["xyxy"], pad=2)
        problem_crop_path = problem_dir / fname(f"problem{idx}", "crop")
        cv2.imwrite(str(problem_crop_path), pb_crop)

        # Stage 2: incremental detect + history
        found_qt, found_choices, img_dets, used_conf_map, last_r2, missing, history_records = detect_incremental_stage2(
            model=model,
            pb_crop=pb_crop,
            qt_id=qt_id,
            choice_ids_map=choice_ids_map,
            img_id=img_id,
            history_dir=h_dir,
            problem_idx=idx
        )

        # problem crop label (ภาพรวม all-classes ที่ conf=0.50)
        all_ids = [qt_id] + list(choice_ids_map.values()) + ([img_id] if img_id is not None else [])
        _, r_vis = yolo_detect(model, pb_crop, all_ids, conf=CONF_STAGE2_START, iou=IOU_STAGE2_NMS, max_det=MAX_DET)
        problem_crop_label_path = add_label_suffix(problem_crop_path)
        cv2.imwrite(str(problem_crop_label_path), r_vis.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE))

        # ถ้าไม่ครบ -> write json + rename + skip
        if missing:
            out_json = {
                "problem_index": idx,
                "status": "INCOMPLETE",
                "missing": missing,
                "page_image": PAGE_IMG,
                "page_pb_label": str(page_pb_label_path.relative_to(OUT_ROOT)),
                "problem_crop": str(problem_crop_path.relative_to(OUT_ROOT)),
                "problem_crop_label": str(problem_crop_label_path.relative_to(OUT_ROOT)),
                "problem_bbox_on_page_xyxy": pb_xyxy_int,
                "used_conf_map": used_conf_map,
                "retry_history": [
                    {
                        "attempt": it["attempt"],
                        "img": str(Path(it["img"]).relative_to(OUT_ROOT)),
                        "json": str(Path(it["json"]).relative_to(OUT_ROOT)),
                    }
                    for it in history_records
                ],
            }
            with open(problem_dir / fname(f"problem{idx}", "result_output", ext=".json"), "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)

            new_dir = rename_dir_incomplete(problem_dir, missing)
            print(f"Skip (incomplete): {new_dir.name}")
            continue

        # ผ่านเงื่อนไขแล้ว
        qt_det = found_qt
        # Question crop
        q_crop, q_xyxy_int = crop_bgr(pb_crop, qt_det["xyxy"], pad=2)
        q_path = q_dir / fname(f"problem{idx}", "question", "crop")
        cv2.imwrite(str(q_path), q_crop)

        saved_question = {
            "path": str(q_path.relative_to(OUT_ROOT)),
            "label_path": str(add_label_suffix(q_path).relative_to(OUT_ROOT)),
            "xyxy_in_problem": q_xyxy_int,
            "conf": qt_det["conf"],
        }

        # latex detect in question (label default)
        lt_dets_q, rlt_q = yolo_detect(model, q_crop, [lt_id], conf=0.25, iou=0.45, max_det=MAX_DET)
        q_label_path = add_label_suffix(q_path)
        cv2.imwrite(str(q_label_path), rlt_q.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE))

        lt_dets_q = sort_boxes_top_left(dedupe_dets_same_class(lt_dets_q, iou_thr=DEDUP_IOU))

        pix2text_question_full = None
        pix2text_question_latex = []

        for j, lt in enumerate(lt_dets_q, start=1):
            lt_crop, _ = crop_bgr(q_crop, lt["xyxy"], pad=2)
            lt_path = l_dir / fname(f"problem{idx}", "qt_lartex", f"{j:02d}")
            cv2.imwrite(str(lt_path), lt_crop)

            if USE_PIX2TEXT:
                out = pix2text_read_image(p2t, bgr_to_pil(lt_crop))
                latex_text = out["latex_list"][0] if out["latex_list"] else out["text"]
                pix2text_question_latex.append({
                    "file": str(lt_path.relative_to(OUT_ROOT)),
                    "latex": latex_text,
                    "raw": out["raw"],
                })

        if USE_PIX2TEXT and PIX2TEXT_READ_FULL_QC:
            pix2text_question_full = pix2text_read_image(p2t, bgr_to_pil(q_crop))

        # Choices crop
        saved_choices = {}
        pix2text_choices_full = {}
        pix2text_choices_latex = {}

        for c in CLS_CHOICES:
            pix2text_choices_latex[c] = []
            det_c = found_choices[c]

            c_crop, c_xyxy_int = crop_bgr(pb_crop, det_c["xyxy"], pad=2)
            c_path = c_dir / fname(f"problem{idx}", c, "crop")
            cv2.imwrite(str(c_path), c_crop)

            saved_choices[c] = {
                "path": str(c_path.relative_to(OUT_ROOT)),
                "label_path": str(add_label_suffix(c_path).relative_to(OUT_ROOT)),
                "xyxy_in_problem": c_xyxy_int,
                "conf": det_c["conf"],
            }

            lt_dets_c, rlt_c = yolo_detect(model, c_crop, [lt_id], conf=0.25, iou=0.45, max_det=MAX_DET)
            c_label_path = add_label_suffix(c_path)
            cv2.imwrite(str(c_label_path), rlt_c.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE))

            lt_dets_c = sort_boxes_top_left(dedupe_dets_same_class(lt_dets_c, iou_thr=DEDUP_IOU))

            for j, lt in enumerate(lt_dets_c, start=1):
                lt_crop, _ = crop_bgr(c_crop, lt["xyxy"], pad=2)
                lt_path = l_dir / fname(f"problem{idx}", f"{c}_lartex", f"{j:02d}")
                cv2.imwrite(str(lt_path), lt_crop)

                if USE_PIX2TEXT:
                    out = pix2text_read_image(p2t, bgr_to_pil(lt_crop))
                    latex_text = out["latex_list"][0] if out["latex_list"] else out["text"]
                    pix2text_choices_latex[c].append({
                        "file": str(lt_path.relative_to(OUT_ROOT)),
                        "latex": latex_text,
                        "raw": out["raw"],
                    })

            if USE_PIX2TEXT and PIX2TEXT_READ_FULL_QC:
                pix2text_choices_full[c] = pix2text_read_image(p2t, bgr_to_pil(c_crop))

        # Save JSON output
        out_json = {
            "problem_index": idx,
            "status": "OK",
            "page_image": PAGE_IMG,
            "page_pb_label": str(page_pb_label_path.relative_to(OUT_ROOT)),
            "problem_crop": str(problem_crop_path.relative_to(OUT_ROOT)),
            "problem_crop_label": str(problem_crop_label_path.relative_to(OUT_ROOT)),
            "problem_bbox_on_page_xyxy": pb_xyxy_int,

            "used_conf_map": used_conf_map,

            "retry_history": [
                {
                    "attempt": it["attempt"],
                    "img": str(Path(it["img"]).relative_to(OUT_ROOT)),
                    "json": str(Path(it["json"]).relative_to(OUT_ROOT)),
                }
                for it in history_records
            ],

            "detections": {
                "question": saved_question,
                "choices": saved_choices,
                "images_in_problem": img_dets,
            },

            "pix2text": {
                "question_full": pix2text_question_full,
                "question_latex": pix2text_question_latex,
                "choices_full": pix2text_choices_full,
                "choices_latex": pix2text_choices_latex,
            },
        }

        with open(problem_dir / fname(f"problem{idx}", "result_output", ext=".json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        print(f"Done OK: {problem_dir}")

    print("All done.")


if __name__ == "__main__":
    main()
