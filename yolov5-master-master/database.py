import cv2
import numpy as np
import time

# Load the YOLOv5 model and its configuration file
# model = cv2.dnn.readNetFromDarknet('C:/yolov5-master-master/yolov5/setup.cfg', 'C:/yolov5-master-master/runs/train/yolov5m_results/weights/best.pt')
model_config = 'yolov5/setup.cfg'
model_weights = 'runs/train/yolov5m_results/weights/best.pt'

# Load the YOLOv5 model
# net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net = cv2.dnn_DetectionModel(model_config, model_weights)

# Set the backend and target for the DNNnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Load the classes file
classes = []
with open('path/to/classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input and output layers of the model
model.setInputSize((640, 640))
model.setInputScale(1.0 / 255)
model.setInputSwapRB(True)

output_layers = model.getUnconnectedOutLayersNames()

# Load the input image
img = cv2.imread('path/to/image.jpg')

# Run the inference
start_time = time.time()
blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (640, 640), [0, 0, 0], True, crop=False)
model.setInput(blob)
outs = model.forward(output_layers)
end_time = time.time()
print('Inference time:', end_time - start_time)

# Parse the output and draw the bounding boxes
conf_threshold = 0.5
nms_threshold = 0.4

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]

    label = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'
    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
    cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
