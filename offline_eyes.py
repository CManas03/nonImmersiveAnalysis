import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0

# Constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# Face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Tracking eye positions
eye_position_counts = {
    "CENTER": 0,
    "PARTIAL": 0,
    "NOT_LOOKING": 0
}

# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean distance function
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)*2 + (y1 - y)*2)
    return distance

# Eyes extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
    return cropped_right, cropped_left

# Position estimator function
def positionEstimator(cropped_eye):
    h, w = cropped_eye.shape
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)
    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)
    return eye_position, color

# Pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    eye_parts = [right_part, center_part, left_part]
    max_index = eye_parts.index(max(eye_parts))
    if max_index == 0:
        pos_eye = "RIGHT"
    elif max_index == 1:
        pos_eye = 'CENTER'
    elif max_index == 2:
        pos_eye = 'LEFT'
    else:
        pos_eye = "CLOSED"
    return pos_eye, []

# Face mesh detection
map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture('/home/cmanas03/Desktop/Nimhans/try.webm')  # Update with the path to your video file

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            eye_position_right, _ = positionEstimator(crop_right)
            eye_position_left, _ = positionEstimator(crop_left)

            # Categorize based on both eye positions
            if eye_position_right == "CENTER" and eye_position_left == "CENTER":
                eye_position_counts["CENTER"] += 1
            elif eye_position_right == "CENTER" or eye_position_left == "CENTER":
                eye_position_counts["PARTIAL"] += 1
            else:
                eye_position_counts["NOT_LOOKING"] += 1

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()

# Plotting the results
positions = list(eye_position_counts.keys())
counts = list(eye_position_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(positions, counts, color=['blue', 'green', 'red'])
plt.xlabel("Eye Position Category")
plt.ylabel("Count")
plt.title("Eye Position Detection Count")
plt.show()