import pymongo
from pymongo import MongoClient
import cv2
import torch

# Define the path to the YOLOv5 model weights file
model_weights_path = "C:/yolov5-master-master/runs/train/yolov5m_results3/weights/best.pt"

# Define the path to the YOLOv5 model configuration file
model_config_path = "path/to/yolov5.yaml"

# Define the input image size that the model was trained on
input_size = (640, 640)

# Define the threshold values for non-maximum suppression
conf_threshold = 0.45
nms_threshold = 0.65

# Define the list of classes that the model was trained on
classes = ['class1', 'class2', 'class3']

# Define the MongoDB connection details
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["detections"]

# Load the YOLOv5 model from the saved weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)
model.yaml_path = model_config_path
model.names = classes
model.eval()

# Load the input image
image = cv2.imread('path/to/image.jpg')

# Resize the image to the input size that the model was trained on
resized_image = cv2.resize(image, input_size)

# Convert the resized image to a PyTorch tensor and normalize its values
input_image = torch.from_numpy(resized_image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

# Pass the input image through the YOLOv5 model to get the detections
with torch.no_grad():
    detections = model(input_image)

# Apply non-maximum suppression to the detections
detections = detections.cpu().numpy()
keep_indices = cv2.dnn.NMSBoxes(detections[:, :4], detections[:, 4], conf_threshold, nms_threshold)

if len(keep_indices) > 0:
    detections = detections[keep_indices.flatten()]

    # Add the detections to the MongoDB database
    for detection in detections:
        if detection is not None:
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            cls_conf, cls_pred = detection[4], detection[5]
            label = classes[int(cls_pred)]
            detection_data = {"class": label, "confidence": cls_conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            collection.insert_one(detection_data)
