import tensorflow as tf
import cv2
import numpy as np
import util
import os

from data.frame_mediapipe import Frame
from data.window import Window
from data.dataAugmentation import Videos
from learning.repetitions import Repetitions


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

    
def classify(model_path):
    
    cap = cv2.VideoCapture(0)
    with util.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor bacj to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = {'x': landmarks[util.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 'y': landmarks[util.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y}
                elbow = {'x': landmarks[util.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 'y': landmarks[util.mp_pose.PoseLandmark.LEFT_ELBOW.value].y}
                wrist = {'x': landmarks[util.mp_pose.PoseLandmark.LEFT_WRIST.value].x, 'y': landmarks[util.mp_pose.PoseLandmark.LEFT_WRIST.value].y}

                # Calculate angle
                angle = util.calculate_angle(shoulder, elbow, wrist)
                print(angle)

                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            except:
                pass

            # Render detections
            util.mp_drawing.draw_landmarks(image, results.pose_landmarks, util.mp_pose.POSE_CONNECTIONS,
                                    util.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    util.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Rilascia la videocamera e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()