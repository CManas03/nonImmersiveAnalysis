import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Variables to track detections
detection_counts = {
    "DETECTED": 0,
    "NOT_DETECTED": 0
}

# List to store detection status at each timestamp
detection_array = []

# Load video from file
video_path = '/home/cmanas03/Desktop/Nimhans/try.webm'  
cap = cv2.VideoCapture(video_path)

# Initialize the Face Detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to capture frame.")
            break

        # Convert the image from BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_frame)

        # Check for detections and classify
        if results.detections:
            detection_counts["DETECTED"] += 1
            detection_array.append([1, 0])  # 1 for "Detected", 0 for "Not Detected"
            # Draw face detection annotations on the image
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        else:
            detection_counts["NOT_DETECTED"] += 1
            detection_array.append([0, 1])  # 0 for "Detected", 1 for "Not Detected"

        # Display the output
        cv2.imshow('MediaPipe Face Detection', frame)

        # Break loop on pressing 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert the detection list to a NumPy array
detection_array = np.array(detection_array)

# Plotting the results
categories = list(detection_counts.keys())
counts = list(detection_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color=['green', 'red'])
plt.xlabel("Detection Status")
plt.ylabel("Count")
plt.title("Face Detection Count per Category")
plt.show()

# Print the numpy array with detections at each timestamp
print("Detection Array:\n", detection_array)