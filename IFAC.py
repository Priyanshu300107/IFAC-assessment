import cv2
import numpy as np
import anthropic
from ultralytics import YOLO

# --- USER INPUT ---
VIDEO_PATH = input("Enter the path to the video file: ")
TRIGGER_EVENTS = input("Enter trigger event keywords (comma separated): ").split(",")
TRIGGER_EVENTS = [event.strip().lower() for event in TRIGGER_EVENTS]
M = int(input("Enter number of frames to include after trigger: "))

# --- YOLOv8 SETUP ---
# Download yolov8n.pt from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
YOLO_MODEL_PATH = "yolov8n.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# Get class names from YOLO model
YOLO_CLASSES = yolo_model.names

client = anthropic.Anthropic(api_key="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")

def detect_objects(frame):
    results = yolo_model(frame)
    detected = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = YOLO_CLASSES[cls_id].lower()
            detected.add(label)
    return list(detected)

def describe_frame(objects):
    prompt = f"Describe the scene in one sentence based on these objects: {', '.join(objects)}."
    response = client.messages.create(
        model="claude-3-haiku-20240229",
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def summarize_event(descriptions):
    prompt = (
        "Summarize the following scene descriptions into a single short sentence:\n"
        + "\n".join(descriptions)
    )
    response = client.messages.create(
        model="claude-3-haiku-20240229",
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# --- MAIN LOGIC ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
event_sequences = []
current_event = []
event_active = False
event_frames_left = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    objects = detect_objects(frame)
    description = describe_frame(objects)

    # Check for trigger event
    if any(trigger in description.lower() for trigger in TRIGGER_EVENTS):
        event_active = True
        event_frames_left = M

    if event_active:
        current_event.append(description)
        event_frames_left -= 1
        if event_frames_left <= 0:
            # Event sequence complete
            summary = summarize_event(current_event)
            event_sequences.append(summary)
            current_event = []
            event_active = False

    frame_idx += 1

cap.release()

# Output summaries
for idx, summary in enumerate(event_sequences):
    print(f"Event {idx+1}: {summary}")
