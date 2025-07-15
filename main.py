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
delayFrames = 20

y_threshold = 300

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

    #converting the inverted normalized value of the y coordinate of the wrist landmark to pixel coordinates
    y_wrist = result["y_wrist"] * height

    print(y_wrist)

    cv2.line(frame, (0, y_threshold), (width, y_threshold), (0, 255, 0), 10)

    if buttonPressed == False and y_wrist > y_threshold:

        if gesture == "next" and imgNumber != len(pathImages) - 1:
            imgNumber += 1
            buttonPressed = True

        if gesture == "previous" and imgNumber != 0:
            imgNumber -= 1
            buttonPressed = True
    
    #frames of delay between inputs ensures that actions don't get triggered in quick succession
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

