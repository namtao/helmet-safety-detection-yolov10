from ultralytics import YOLO
from pytube import YouTube
import cv2

# Download the YouTube video
YOUTUBE_VIDEO_URL = "https://youtu.be/wqPSsu7XQ74"
yt = YouTube(YOUTUBE_VIDEO_URL)
stream = yt.streams.filter(file_extension="mp4").first()
video_path = "youtube_video.mp4"
stream.download(filename=video_path)

# Load the YOLO model
MODEL_PATH = r"../yolov10/yolov10n.pt"
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on each frame
    results = model(frame)

    # Draw the detection results on the frame
    for result in results:
        frame = result.plot()

    # Write the frame to the output video
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
