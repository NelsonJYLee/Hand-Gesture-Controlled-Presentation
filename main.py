import cv2
import os

#variables
width, height = 1280, 720
folderPath = "Presentations"

#camer setup
cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)

#Get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key = len)
print(pathImages)

imgNumber = 0
hs, ws = int(120*1.2), int(213*1.2)

while True:
    #Import Images
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Adding webcam image on slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


