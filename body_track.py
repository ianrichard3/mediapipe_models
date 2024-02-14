import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



# Video Feed

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        
        


        # Detect stuff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        # To re-render the image in bgr format
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # print(results.pose_landmarks)

        
        # if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                     mp_pose.POSE_CONNECTIONS)
        



        cv2.imshow("Mediapipe feed", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


    cap.release()

    cv2.destroyAllWindows()