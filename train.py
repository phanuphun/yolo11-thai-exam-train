from ultralytics import YOLO

model = YOLO("best.pt")  # load a pretrained YOLOv8n model

train_results = model.train(
    data="datasets/data.yaml",  
    epochs=100, 
    imgsz=640,
    device="0",
    workers=0,
    val=True,
)

# metrics = model.val()