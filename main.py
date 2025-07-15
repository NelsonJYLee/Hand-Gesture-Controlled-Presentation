import cv2
import os

from classifier import Classifier


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

classifier = Classifier()

buttonPressed = False
pressedFrames = 0
delayFrames = 10

while True:
    #Import Images
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Adding webcam image on slides
    imgSmall = cv2.resize(frame, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    result = classifier.new_frame(frame)
    frame = result["frame"]
    gesture = result["predicted gesture"]
    confidence = result["confidence"]

    #now I integrate moving the powerpoint with the gestures

    if buttonPressed == False:

        if gesture == "next" and imgNumber != len(pathImages) - 1:
            imgNumber += 1
            buttonPressed = True

        if gesture == "previous" and imgNumber != 0:
            imgNumber -= 1
            buttonPressed = True
        
    if buttonPressed:
        pressedFrames += 1
        if pressedFrames > delayFrames:
            pressedFrames = 0
            buttonPressed = False
    
    


    cv2.imshow("Image", frame)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

