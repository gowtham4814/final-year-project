import torch
import cv2
from collections import Counter
import pymongo
import time
import datetime
# Set the time interval for inserting the detected classes and their counts
interval = 5  # in seconds
last_insert_time = time.time()
# Load the pre-trained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='C:/yolov5-master-master/runs/train/yolov5m_results3/weights/best.pt')
# Set the model to evaluation mode
model.eval()

DATABASE_NAME = 'dummy'
DATABASE_USERNAME = 'gowtham'
DATABASE_PASSWORD = 'tbbjOmjorm0wDwK1'
DATABASE_CLUSTER = 'cluster0.88yorqa.mongodb.net'
DATABASE_URI = f"mongodb+srv://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_CLUSTER}/{DATABASE_NAME}?retryWrites=true&w=majority"

client = pymongo.MongoClient(DATABASE_URI)
db = client['dummy']
collection = db['detects']
# Define the classes you want to detect
classes = ['safe', 'no_mask', 'no_helmet', 'not_safe', 'fire', 'unconcious']

# Set the minimum confidence level for detections
conf_thresh = 0.005
one_day_ago = datetime.datetime.now() - datetime.timedelta(hours=24)

# Delete documents inserted 24 hours ago
result = collection.delete_many({'detected_classes.timestamp': {'$lt': one_day_ago}})
print(f"Deleted {result.deleted_count} documents")
# Open the camera

cap = cv2.VideoCapture('http://10.121.19.171:8080/video')
# cap = cv2.VideoCapture(0)
# Set the resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Loop over the frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects in the frame
    results = model(img)

    # Get the detection results
    detections = results.pandas().xyxy[0]

    # Loop over the detections and draw bounding boxes around the objects
    detected_classes = []
    for _, detection in detections.iterrows():
        x1, y1, x2, y2, conf = detection[0:5]
        class_id = int(detection[5])
        if conf > conf_thresh:
            color = (0, 191, 255)  # Green color for bounding boxes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{classes[class_id]}: {conf:.2f}"
            detected_classes.append(classes[class_id])
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output frame
    cv2.imshow('Output', frame)
    current_time = time.time()
    print(detected_classes)
    if current_time - last_insert_time > interval:
        last_insert_time = current_time
    # Print the names of all the detected classes

        if detected_classes:
            # Define the class names to be tracked
            class_names = ['safe', 'no_mask', 'not_safe', 'unconcious', 'no_helmet', 'fire']

            # Get the class counts
            class_counts = Counter(detected_classes)

            # Initialize a dictionary to hold the class counts
            class_counts_dict = {}

            # Set the count for each class to 0 by default
            for class_name in class_names:
                class_counts_dict[class_name] = 0

            # Update the dictionary with the actual counts
            for class_name, count in class_counts.items():
                class_counts_dict[class_name] = count

            # Insert the class counts into the database
            unix_timestamp = datetime.datetime.now().timestamp()
            data = {
                'safe': class_counts_dict['safe'],
                'no_mask': class_counts_dict['no_mask'],
                'not_safe': class_counts_dict['not_safe'],
                'unconcious': class_counts_dict['unconcious'],
                'no_helmet': class_counts_dict['no_helmet'],
                'fire': class_counts_dict['fire'],
                'timestamp': datetime.datetime.fromtimestamp(unix_timestamp)
            }
            result = collection.insert_one(data)
            print(f"Inserted new document with ID {result.inserted_id}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

