import cv2
import time
import numpy as np

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"
)

zones = [
    [1, 414, 805, 845],    # x1 < x2, y1 < y2
    [1, 332, 487, 463],   # x1 < x2, y1 < y2
    [920, 320, 1600, 480] # x1 < x2, y1 < y2
]

cap = cv2.VideoCapture("https://cams.is74.ru/live/main/cam1026.m3u8")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for (x1, y1, x2, y2) in zones:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(roi, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                # ! только нужные классы:
                if label not in ["car", "bus", "train"]:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([roi.shape[1], roi.shape[0], roi.shape[1], roi.shape[0]])
                (rx1, ry1, rx2, ry2) = box.astype("int")
                cv2.rectangle(frame, (rx1 + x1, ry1 + y1), (rx2 + x1, ry2 + y1), (0, 255, 0), 2)
                cv2.putText(frame, label, (rx1 + x1, ry1 + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


    cv2.imshow("Detection", frame)

    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
