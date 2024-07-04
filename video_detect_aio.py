from ultralytics import YOLOv10

MODEL_PATH = r"../yolov10/yolov10n.pt"
model = YOLOv10(MODEL_PATH)

YOUTUBE_VIDEO_PATH = "https://youtu.be/wqPSsu7XQ74"
video_result = model(source=YOUTUBE_VIDEO_PATH)
