import cv2
import time
import numpy as np

# COCO классы (YOLOv5 обучен на COCO dataset)
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
          'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
          'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Загружаем YOLOv5 ONNX модель
net = cv2.dnn.readNet("yolov5s.onnx")

zones = [
    [1, 554, 377, 826],  # x1 < x2, y1 < y2
    [350, 422, 598, 693],  # x1 < x2, y1 < y2
    [567, 454, 796, 642],
    [714, 395, 983, 603],
    [1, 294, 126, 410],
    [78, 325, 241, 456],
    [145, 395, 384, 552],
    [390, 395, 1073, 474],
    [145, 378, 384, 552],
    [1075, 366, 1278, 440],
    [1276, 343, 1441, 404],
    [966, 490, 1130, 624],
    [1086, 552, 1276, 672],
    [1273, 606, 1442, 720],
    [1402, 662, 1590, 776],
    [1541, 686, 1829, 818],
]
def is_in_zone(x1, y1, x2, y2, zones):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    for zx1, zy1, zx2, zy2 in zones:
        if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
            return True
    return False

cap = cv2.VideoCapture("https://cams.is74.ru/live/main/cam1026.m3u8")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 принимает вход 640x640
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    # Парсим результаты YOLOv5
    for zx1, zy1, zx2, zy2 in zones:
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # порог уверенности
                label = CLASSES[class_id]
                if label not in ["car", "bus", "truck", "motorcycle"]:
                    continue
                    
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                
                x1 = int(center_x - width/2)
                y1 = int(center_y - height/2)
                x2 = int(center_x + width/2)
                y2 = int(center_y + height/2)
                
                if is_in_zone(x1, y1, x2, y2, zones):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(frame, (center_x, y1-20), (center_x, y1), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.imshow("YOLOv5 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
