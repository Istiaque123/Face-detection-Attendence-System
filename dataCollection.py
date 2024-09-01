import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

###################################################
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
debug = False


confidence = 0.8
save = True
blurThreshold = 35 # Larger is more focus

classID = int(input("Enter '0' for fake data, '1' for real data & '2' for store any: ")) # 0 is fake & 1 is real

if classID == 0:
    outputFolderPath = 'Dataset/Fake Data'
elif classID == 1:
    outputFolderPath = 'Dataset/Real Data'
else:
    outputFolderPath ='Dataset/DataCollect'
###################################################


cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()

    img, bboxs = detector.findFaces(img, draw = False)

    listBlur = [] # true False values indicating if the faces are blur or not
    listInfo = [] # the normalized values and the class name for the label txt file


    if bboxs:
        for bbox in bboxs:

            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox["score"][0]
            # print(score)
            # print(x, y, w, h)

            # ----------- Check the Score ---------
            if score > confidence:

                # ------- Adding an offset to the face Detected--------------
                # Increasing offset width

                offsetW = (offsetPercentageW / 100) * w

                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                # Increasing offset height

                offsetH = (offsetPercentageH / 100) * h

                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3)

                # ------ to avoid values below 0 --------

                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ---------------- Adding Bluriness on Face --------------

                imgFace = img[y: y + h, x: x + w]

                cv2.imshow("Face", imgFace) # this is use for capture img for training

                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)


                # ---------------- Normalization --------------
                iH, iW, _ = img.shape

                xC, yC = x + w / 2, y + h / 2

                xCN = round(xC / iW, floatingPoint)
                yCN = round(yC / iH, floatingPoint)

                wN = round(w / iW, floatingPoint)
                hN = round(h / iH, floatingPoint)
                # print(xCN, yCN, wN, hN)

                # ---------avoid value above 1 ------
                if xCN < 0: xCN = 1
                if yCN < 0: yCN = 1
                if wN < 0:  wN = 1
                if hN < 0:  hN = 1

                listInfo.append(f"{classID} {xCN} {yCN} {wN} {hN}\n")

                # ---------------- Drawing --------------

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)

                cvzone.putTextRect(imgOut, f'Score : {int(score * 100)}% Blur: {blurValue}', (x, y - 0), scale= 2, thickness= 4)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)

                    cvzone.putTextRect(img, f'Score : {int(score * 100)}% Blur: {blurValue}', (x, y - 0), scale= 2, thickness= 4)
                # cvzone.cornerRect(img, (x, y, w, h))


                # ---------------- To save --------------

            if save :
                # print(listBlur, all(listBlur))
                if all(listBlur) and listBlur != []:
                # ---------------- Save Img --------------
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0]+timeNow[1]
                    print(timeNow)

                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

                # ---------------- Save Label Text File --------------

                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()



    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
