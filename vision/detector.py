from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects_frame(frame, return_positions=False):
    results = model.predict(frame)
    detected = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x_center = float(box.xywh[0][0])  # center x coordinate
            if return_positions:
                detected.append((label, x_center))
            else:
                detected.append(label)
    return detected
