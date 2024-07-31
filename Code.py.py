import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from datetime import datetime

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change to 0 if 1 doesn't work
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a figure and axes
fig, ax = plt.subplots(1, 1)
plt.ion()  # Turn on interactive mode

# Function to calculate centroid of a bounding box
def get_centroid(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

# List to store centroids and detected persons
centroids = []

# Distance threshold to determine if a centroid matches an existing person
distance_threshold = 50

# Function to append a detection to the JSON file
def append_detection(detection, filename='detections.json'):
    try:
        with open(filename, 'r+') as file:
            data = json.load(file)
            data.append(detection)
            file.seek(0)
            json.dump(data, file, indent=4, default=str)
    except FileNotFoundError:
        with open(filename, 'w') as file:
            json.dump([detection], file, indent=4, default=str)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform object detection
    results = model(frame)

    # Clear the previous image and bounding boxes
    ax.clear()

    # Convert frame to RGB format for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    ax.imshow(frame_rgb)

    # Temporary list to store current frame centroids
    current_centroids = []

    # Draw bounding boxes for persons only
    for result in results:
        for detection in result.boxes:
            if detection.cls[0] == 0 and detection.conf[0] > 0.5:  # Class ID for persons is 0
                # Get the coordinates and dimensions of the bounding box
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                width, height = x2 - x1, y2 - y1

                # Calculate centroid
                centroid = get_centroid(x1, y1, x2, y2)
                current_centroids.append(centroid)

                # Create a rectangle patch
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Add label
                label = f"Person {detection.conf[0]:.2f}"
                ax.text(x1, y1, label, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Update centroids list with current frame centroids
    if len(centroids) == 0:
        centroids = current_centroids
        for c in current_centroids:
            append_detection({"timestamp": datetime.now()})
    else:
        for c in current_centroids:
            distances = np.linalg.norm(np.array(centroids) - np.array(c), axis=1)
            if np.all(distances > distance_threshold):
                centroids.append(c)
                append_detection({"timestamp": datetime.now()})

    # Display the number of unique persons detected
    num_persons = len(centroids)
    ax.text(10, 30, f"Number of persons detected: {num_persons}", color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # Update the plot
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

    # Check for 'q' key press to exit
    if plt.waitforbuttonpress(0.001):
        break

# Release the capture and close the figure
cap.release()
plt.close(fig)