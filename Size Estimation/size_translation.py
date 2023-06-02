import cv2
import cvzone
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def closest_value(input_list, input_value):

    arr = np.asarray(input_list)

    i = (np.abs(arr - input_value)).argmin()

    return arr[i]


def hipstoEU(size_hips):
    if size_hips == 90:
        size_hips_EU = 42
    if size_hips == 93:
        size_hips_EU = 44
    if size_hips == 96:
        size_hips_EU = 46
    if size_hips == 99:
        size_hips_EU = 48
    if size_hips == 102:
        size_hips_EU = 50
    if size_hips == 105:
        size_hips_EU = 52
    if size_hips == 108:
        size_hips_EU = 54
    if size_hips == 111:
        size_hips_EU = 56
    if size_hips == 114:
        size_hips_EU = 58
    if size_hips == 117:
        size_hips_EU = 60
    if size_hips == 120:
        size_hips_EU = 62
    if size_hips == 123:
        size_hips_EU = 64
    if size_hips == 126:
        size_hips_EU = 66
    if size_hips == 129:
        size_hips_EU = 68
    if size_hips == 132:
        size_hips_EU = 70
    if size_hips == 135:
        size_hips_EU = 72
    if size_hips == 138:
        size_hips_EU = 74
    if size_hips == 143:
        size_hips_EU = 76
    if size_hips == 147:
        size_hips_EU = 78
    return size_hips_EU


def size_to_name(recommended_size):
    if recommended_size == 42:
        size_text = "XS"
    if recommended_size == 44:
        size_text = "S"
    if recommended_size == 46:
        size_text = "S"
    if recommended_size == 48:
        size_text = "M"
    if recommended_size == 50:
        size_text = "M"
    if recommended_size == 52:
        size_text = "L"
    if recommended_size == 54:
        size_text = "XL"
    if recommended_size == 56:
        size_text = "XXL"
    if recommended_size == 58:
        size_text = "3XL"
    if recommended_size == 60:
        size_text = "3XL"
    if recommended_size == 62:
        size_text = "4XL"
    if recommended_size == 64:
        size_text = "4XL"
    if recommended_size == 66:
        size_text = "5XL"
    if recommended_size == 68:
        size_text = "5XL"
    if recommended_size == 70:
        size_text = "6XL"
    if recommended_size == 72:
        size_text = "6XL"
    if recommended_size == 74:
        size_text = "7XL"
    if recommended_size == 76:
        size_text = "8XL"
    if recommended_size == 78:
        size_text = "8XL"
    return size_text


def size_estimation():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    success, image = cap.read()

    imgFront = cv2.imread("outline.png", cv2.IMREAD_UNCHANGED)
    imgFront = cv2.resize(imgFront, (0, 0), None, 0.3, 0.3)

    hf, wf, cf = imgFront.shape
    hb, wb, cb = image.shape

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            list_chest = [84, 88, 92, 96, 100, 104, 108, 112, 116,
                          120, 124, 128, 132, 136, 140, 144, 148, 152, 156]

            list_hips = [90, 93, 96, 99, 102, 105, 108, 111, 114,
                         117, 120, 123, 126, 129, 132, 135, 138, 143, 147]

            list_sizes = [42, 44, 46, 48, 50, 52, 54, 56,
                          58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78]

            success, image = cap.read()
            imgResult = cvzone.overlayPNG(image, imgFront, [200, 6])
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Extract landmarks
            try:
                global landmarks
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            shoulder_L = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            shoulder_R = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            hip_L = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
            hip_R = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x

            distance_chest = shoulder_L - shoulder_R
            distance_hip = hip_L - hip_R
            chest = (distance_chest * 167) * 2 + 34
            hip = (distance_hip * 167) * 2 + 43

            size_chest = closest_value(list_chest, chest)
            size_hips = closest_value(list_hips, hip)

            size_hips_EU = hipstoEU(size_hips)

            average_size = (((size_chest/2) + 2) + size_hips_EU) / 2

            recommended_size = closest_value(list_sizes, average_size)

            size_text = size_to_name(recommended_size)

            print((size_chest/2) + 2)
            print(size_hips_EU)

            # Setup status box
            color = (255, 255, 255)
            cv2.rectangle(imgResult, (0, 0), (250, 73),
                          (0, 0, 0), -1)

            # Rep data
            cv2.putText(imgResult, "EU Size", (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(imgResult, str(recommended_size) + " " + size_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', imgResult)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
