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
#sort the files by numerical names
pathImages = sorted(
    [f for f in os.listdir(folderPath) if f.endswith((".jpg", ".png"))],
    key=lambda x: int(os.path.splitext(x)[0])
)
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
    pointer_coord = result["pointer_coord"]

    #converting normalized pointer finger x y values to pixel coordinates, and then shifting x by 0.5w and adding sensitivity of 2.2
    new_X = int(int(int(pointer_coord[0] * w) - 0.5*w) * 2.2)
    
    pointer_coord_pixel = (new_X, int(pointer_coord[1] * h))

    #converting the inverted normalized value of the y coordinate of the wrist landmark to pixel coordinates
    y_wrist = result["y_wrist"] * height

    cv2.line(frame, (0, y_threshold), (width, y_threshold), (0, 255, 0), 10)

    if buttonPressed == False and y_wrist > y_threshold:

        #Gesture 1: show next slide
        if gesture == "next" and imgNumber != len(pathImages) - 1:
            imgNumber += 1
            buttonPressed = True

        #Gesture 2: show previous slide
        if gesture == "previous" and imgNumber != 0:
            imgNumber -= 1
            buttonPressed = True

    #need to figure out a way to get more range with my hand movements,
    #Gesture3: pointer on slide
    if gesture == "pointer":
        cv2.circle(imgCurrent, pointer_coord_pixel, 12, (0, 0, 225), cv2.FILLED)
    
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

