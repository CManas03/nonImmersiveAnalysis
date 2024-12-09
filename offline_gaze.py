import numpy as np
import cv2
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

# Load video from file
video_path = '/home/cmanas03/Desktop/Nimhans/try.webm'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Initialize an empty list to store data
data = []

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or failed to capture frame.")
            break

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # Flipped for selfie view
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in {33, 263, 1, 61, 291, 199}:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append(([x, y, lm.z]))

                # Get 2D and 3D coordinates
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # Get rotational angles
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                z_angle = angles[2] * 360

                # Define orientation
                if abs(y_angle) <= 10 and abs(x_angle) <= 10:
                    orientation = "Forward"
                else:
                    orientation = "Not Forward"

                # Store data
                data.append([x_angle, y_angle, z_angle, orientation])

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)
                cv2.putText(image, orientation, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            print("FPS: ", fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)

        cv2.imshow('Head Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

except KeyboardInterrupt:
    print("Program interrupted. Saving data...")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Save data as a numpy array
    data_array = np.array(data, dtype=object)
    np.save('head_pose_data.npy', data_array)
    print("Data saved successfully as a NumPy array!")

    # Save data as a CSV file
    csv_file_path = '/home/cmanas03/Desktop/Nimhans/head_pose_data.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y', 'Z', 'Orientation'])
        writer.writerows(data)
    print("Data saved successfully as a CSV file!")

    # Plot orientation counts
    orientations = [row[3] for row in data]
    orientation_counts = {"Forward": orientations.count("Forward"), "Not Forward": orientations.count("Not Forward")}

    plt.figure(figsize=(10, 6))
    plt.bar(orientation_counts.keys(), orientation_counts.values(), color=['blue', 'orange'])
    plt.xlabel("Orientation")
    plt.ylabel("Count")
    plt.title("Head Pose Orientation Counts")
    plt.show()