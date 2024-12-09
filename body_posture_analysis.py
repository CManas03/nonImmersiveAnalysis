import cv2
import csv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize Mediapipe Pose and Drawing Utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load video from file
video_path = '/home/cmanas03/Desktop/Nimhans/posture.webm'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Thresholds
POSTURE_THRESHOLD = 0.15  # Threshold for front-facing vs. side-facing
FORWARD_POSTURE_THRESHOLD = 0.35  # Threshold for detecting forward posture (z-depth)
SIDE_FORWARD_POSTURE_THRESHOLD = 0.15  # Threshold for detecting forward posture (x-depth)

# Array to store posture counts
posture_counts = {
    "Good Posture": 0,
    "Forward Posture": 0,
    "Side Facing": 0,
    "Front Facing": 0
}

# List to store position data
position_data = []

# Pose estimation using Mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to capture frame.")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process image and extract pose landmarks
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and analyze posture
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x

            head_z = landmarks[mp_pose.PoseLandmark.NOSE.value].z
            left_elbow_z = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z
            right_elbow_z = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
            avg_elbow_z = (left_elbow_z + right_elbow_z) / 2

            nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
            avg_elbow_x = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x) / 2

            # Determine if the posture is front-facing or side-facing
            shoulder_diff_x = abs(left_shoulder_x - right_shoulder_x)
            posture_orientation = "Side Facing" if shoulder_diff_x < POSTURE_THRESHOLD else "Front Facing"
            posture_counts[posture_orientation] += 1

            if posture_orientation == "Side Facing":
                nose_shoulder_diff_x = abs(nose_x - avg_elbow_x)
                posture_depth = "Good Posture" if nose_shoulder_diff_x < SIDE_FORWARD_POSTURE_THRESHOLD else "Forward Posture"
                posture_status = f"Side Facing - {posture_depth}"
            else:
                # Determine if the posture is forward-leaning
                avg_depth = abs(head_z - avg_elbow_z)
                posture_depth = "Good Posture" if avg_depth < FORWARD_POSTURE_THRESHOLD else "Forward Posture"
                posture_status = f"Front Facing - {posture_depth}"

            posture_counts[posture_depth] += 1

            # Save position data along with posture orientation and depth
            position_data.append([
                left_shoulder_x, right_shoulder_x, head_z, left_elbow_z, right_elbow_z, avg_elbow_z, nose_x, avg_elbow_x,
                posture_orientation, posture_depth
            ])

            # Display posture status on the image
            cv2.putText(image, posture_status,
                        (50, 50),  # Text position
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if posture_depth == "Good Posture" else (0, 0, 255),
                        2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Posture Analysis', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

position_data_array = np.array(position_data)

# Convert numpy array to list of lists and ensure all elements are strings
position_data_list = position_data_array.astype(str).tolist()

# Save the position data to a CSV file
csv_file = "posture_data.csv"
csv_columns = ["left_shoulder_x", "right_shoulder_x", "head_z", "left_elbow_z", "right_elbow_z", "avg_elbow_z", "nose_x", "avg_elbow_x", "posture_orientation", "posture_depth"]

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
        writer.writerows(position_data_list)
except IOError:
    print("I/O error")
# Generate graphs
labels = ['Good Posture', 'Forward Posture']
values = [posture_counts['Good Posture'], posture_counts['Forward Posture']]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(labels, values, color=['green', 'red'])
plt.title('Posture Depth Analysis')
plt.xlabel('Posture')
plt.ylabel('Count')

labels = ['Side Facing', 'Front Facing']
values = [posture_counts['Side Facing'], posture_counts['Front Facing']]

plt.subplot(1, 2, 2)
plt.bar(labels, values, color=['blue', 'orange'])
plt.title('Posture Orientation Analysis')
plt.xlabel('Posture')
plt.ylabel('Count')

plt.tight_layout()
plt.show()