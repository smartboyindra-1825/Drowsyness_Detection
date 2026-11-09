import streamlit as st
import cv2
import dlib
import numpy as np
import time
import threading
import os

from scipy.spatial import distance as dist
from imutils import face_utils
from playsound import playsound


# -----------------------------
# Global alarm status
# -----------------------------
alarm_status = False


# -----------------------------
# Alarm function
# -----------------------------
def alarm(msg):
    global alarm_status

    while alarm_status:
        try:
            playsound(r"C:\New folder\alarm.wav")
        except Exception as e:
            st.warning(f"Error playing sound: {e}")
            break

        time.sleep(1)


# -----------------------------
# Eye Aspect Ratio
# -----------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


# -----------------------------
# Calculate EAR
# -----------------------------
def final_ear(shape):

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    return ear, leftEye, rightEye


# -----------------------------
# Calculate lip distance
# -----------------------------
def lip_distance(shape):

    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])

    return distance


# -----------------------------
# Thresholds
# -----------------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
YAWN_THRESH = 35


# -----------------------------
# Main detection function
# -----------------------------
def live_drowsiness_detection():

    global alarm_status

    st.title("Live Drowsiness and Yawn Detection")

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Unable to access webcam.")
        return

    frame_placeholder = st.empty()

    # Face detector
    detector = dlib.get_frontal_face_detector()

    # IMPORTANT:
    # Change this path if your .dat file is somewhere else.
    predictor_path = r"C:\New folder\shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(predictor_path):
        st.error(f"Landmark file not found:\n{predictor_path}")
        video_capture.release()
        return

    predictor = dlib.shape_predictor(predictor_path)

    COUNTER = 0
    alarm_status = False

    while True:

        ret, frame = video_capture.read()

        if not ret:
            st.warning("Lost connection to webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate EAR
            ear, leftEye, rightEye = final_ear(shape)

            # Calculate yawn distance
            distance = lip_distance(shape)

            # Eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(
                frame,
                [leftEyeHull],
                -1,
                (0, 255, 0),
                1
            )

            cv2.drawContours(
                frame,
                [rightEyeHull],
                -1,
                (0, 255, 0),
                1
            )

            # Mouth contour
            lip = shape[48:60]

            cv2.drawContours(
                frame,
                [lip],
                -1,
                (0, 255, 0),
                1
            )

            # -----------------------------
            # Drowsiness detection
            # -----------------------------
            if ear < EYE_AR_THRESH:

                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    if not alarm_status:

                        alarm_status = True

                        t = threading.Thread(
                            target=alarm,
                            args=("Wake up sir",)
                        )

                        t.daemon = True
                        t.start()

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            else:

                COUNTER = 0
                alarm_status = False

            # -----------------------------
            # Yawn detection
            # -----------------------------
            if distance > YAWN_THRESH:

                cv2.putText(
                    frame,
                    "Yawn Alert",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # -----------------------------
            # Display EAR
            # -----------------------------
            cv2.putText(
                frame,
                "EAR: {:.2f}".format(ear),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # -----------------------------
            # Display Yawn value
            # -----------------------------
            cv2.putText(
                frame,
                "YAWN: {:.2f}".format(distance),
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Show frame in Streamlit
        frame_placeholder.image(
            frame,
            channels="BGR"
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    video_capture.release()

    alarm_status = False

    cv2.destroyAllWindows()


# -----------------------------
# Start application
# -----------------------------
if __name__ == "__main__":
    live_drowsiness_detection()
