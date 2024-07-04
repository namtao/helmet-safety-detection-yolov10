from ultralytics import YOLOv10

MODEL_PATH = r"../yolov10/yolov10n.pt"
model = YOLOv10(MODEL_PATH)

IMG_PATH = r"./images/HCMC_Street.jpg"
result = model(source=IMG_PATH)[0]
result.save(r"./images/HCMC_Street_predict.png")
