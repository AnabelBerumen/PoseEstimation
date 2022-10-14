import cv2
import mediapipe as mp #rgb
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# read the video
cap = cv2.VideoCapture('PoseVideos/2.mp4')
pTime = 0
while True:
    # this img is in BGR
    success, img = cap.read()
    # convert the image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # height width chanel
            h, w, c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print the circle on top of this point
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)