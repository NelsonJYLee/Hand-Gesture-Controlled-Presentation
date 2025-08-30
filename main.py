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
hs, ws = 0, 0

classifier = Classifier()

buttonPressed = False
pressedFrames = 0
delayFrames = 10

y_threshold = 400

#list of lists, each reprsenting the annotations for each slide
#each slide list will have lists of annotation groups, seperating the coordinates that need to be linked
#each list of annotation groups will contain sets of xy coordinates that will be linked
annotations = [[] for _ in range(len(pathImages))]

annotation_start = False
annotation_number = [-1 for _ in range(len(pathImages))]

while True:
    #Import Images
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Adding webcam image on slides
    h, w, _ = imgCurrent.shape
    hs = int(h * 0.25)
    ws = int (w * 0.25)
    imgSmall = cv2.resize(frame, (ws, hs))
    imgCurrent[0:hs, w - ws:w] = imgSmall

    result = classifier.new_frame(frame)
    frame = result["frame"]
    gesture = result["predicted gesture"]
    print(gesture)
    confidence = result["confidence"]
    pointer_coord = result["pointer_coord"]

    #converting normalized pointer finger x y values to pixel coordinates

    #shifting x by 0.5w and adding sensitivity of 2.2
    new_X = int(int(int(pointer_coord[0] * w) - 0.5*w) * 2.2)
    #shifting y by 0.25h and adding sensitivity of 2.2
    new_Y = int(int(int(pointer_coord[1] * h) - 0.25*h) * 2.2)
    
    pointer_coord_pixel = (new_X, new_Y)

    #converting the inverted normalized value of the y coordinate of the wrist landmark to pixel coordinates
    y_wrist = result["y_wrist"] * height

    cv2.line(frame, (0, y_threshold), (width, y_threshold), (128, 0, 0), 10)

    if buttonPressed == False and y_wrist < y_threshold:

        #Gesture 1: show next slide
        if gesture == "next" and imgNumber != len(pathImages) - 1:
            imgNumber += 1
            buttonPressed = True

        #Gesture 2: show previous slide
        if gesture == "previous" and imgNumber != 0:
            imgNumber -= 1
            buttonPressed = True

    #Gesture3: pointer on slide
    if gesture == "pointer":
        cv2.circle(imgCurrent, pointer_coord_pixel, 16, (0, 0, 225), cv2.FILLED)

    #Gesture4: draw on slide
    if gesture == "drawer":
        #creates a new list of coordinates
        if annotation_start == False:
            annotation_start = True
            annotation_number[imgNumber] += 1
            annotations[imgNumber].append([])
        cv2.circle(imgCurrent, pointer_coord_pixel, 16, (0, 0, 225), cv2.FILLED)
        annotations[imgNumber][annotation_number[imgNumber]].append(pointer_coord_pixel)
    else:
        annotation_start = False

    #Gesture5: erase last drawing
    if gesture == "erase" and not buttonPressed:
        if annotations[imgNumber]:
            annotations[imgNumber].pop()
            annotation_number[imgNumber] -= 1
            buttonPressed = True
    
    #connect the coordinates of each list in each slide
    for i in range(len(annotations[imgNumber])):
        for j in range(len(annotations[imgNumber][i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[imgNumber][i][j-1], annotations[imgNumber][i][j], (0,0,200), 16)
    
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