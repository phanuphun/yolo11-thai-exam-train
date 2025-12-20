from ultralytics import YOLO

model = YOLO("best.pt")
name_to_id = {v: k for k, v in model.names.items()}

keep = [name_to_id["lt"]] 
# keep = [name_to_id["pb"]] 
# keep = [name_to_id["c"]] 
# keep = [name_to_id["pb"],name_to_id["qt"], name_to_id["c"]] 
keep = [name_to_id["qt"]] 
# results = model.predict("test_data/12.png",)
results = model.predict(
    "test_data/13.png", 
    classes=keep,
    conf=0.1,
    iou=0.45,
    agnostic_nms=False,
    augment=False,
)
# print(results[0].boxes.cls)
print(results[0].boxes.xyxy)  # bounding boxes in xyxy format
# print(results[0].boxes.xywh)  # bounding boxes in xywh format
results[0].show(font_size=2, line_width=1)
