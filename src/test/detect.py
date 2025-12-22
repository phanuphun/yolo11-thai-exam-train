from ultralytics import YOLO

model = YOLO("src/models/best.pt")
name_to_id = {v: k for k, v in model.names.items()}
keep = [name_to_id["pb"], name_to_id["qt"], name_to_id["cg"], name_to_id["c"], name_to_id["img"]] 
results = model.predict(
    "test_data/vol0/13.png", 
    classes=keep,
    conf=0.1,
    iou=0.45,
    agnostic_nms=False,
    augment=False,
)
print(results[0].boxes.xyxy)  # bounding boxes in xyxy format
results[0].show(font_size=2, line_width=1)
