import numpy as np
import cv2

# Creating a VideoCapture object.
cap = cv2.VideoCapture(0)

# Getting Background.
for i in range(10):     # Throwing away the first 9 frames.
    success, frame = cap.read()
if not success:
    exit(1)

gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Face detection and hiding it on background reference.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face = face_cascade.detectMultiScale(gray_background, 1.1, 4)
height = gray_background.shape[1]
if face:
    for (x1, y1, w1, h1) in face:
        gray_background[y1:height, x1:x1 + w1] = 0

# Kernels Settings.
K_SIZE = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Applying Blur on background reference.
gray_background = cv2.GaussianBlur(gray_background, (K_SIZE, K_SIZE), 0)

# Inicial Settings.
captura = False
index = 1


while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        exit(1)

    # Putting a black rectangle on face position.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame[y1:height, x1:x1 + w1] = 0

    # Applying Blur
    gray_frame = cv2.GaussianBlur(gray_frame, (K_SIZE, K_SIZE), 0)

    # Difference between background reference and current Frame
    moviment = cv2.absdiff(gray_background, gray_frame)

    binary = cv2.threshold(moviment, 40, 255, cv2.THRESH_BINARY)[1]
    cv2.erode(binary, erode_kernel, binary, iterations=2)
    cv2.dilate(binary, dilate_kernel, binary, iterations=2)

    # Finding contours.
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Storing contours with areas > 4000 and counting them.
    figures = []
    for cont in contours:
        if cv2.contourArea(cont) > 4000:
            figures.append(cont)
    figures.sort(key=cv2.contourArea, reverse=True)
    n_figures = len(figures)

    # Saving hand images
    if cv2.waitKey(1) == ord('c'):
        captura = True
        print("SALVANDO .................")


    if n_figures > 0:
        # Drawing blue rectangle around bigger figure.
        x, y, w, h = cv2.boundingRect(figures[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Region of interest
        hand_ROI = frame[y:y+h, x:x+w]
        hand_ROIP = np.zeros(hand_ROI.shape, np.uint8)
        if hand_ROI.shape[0] > 0:
            mask = binary[y:y+h, x:x + w] > 200
            hand_ROIP[mask] = hand_ROI[mask]

        # Saving dataset images    
        if captura:
            if index == 201:
                captura = False
                break
            figSave = hand_ROIP.copy()
            figSave = cv2.resize(figSave, (256, 256))
            cv2.imwrite("img"+str(index)+".png", figSave)
            index += 1

    
    cv2.imshow('detection', frame)

    k = cv2.waitKey(1)
    if k == 27:  # Esc.
        break
    

cap.release()
cv2.destroyAllWindows()
