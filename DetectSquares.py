import cv2
import time
import numpy as np
import cv2.aruco as aruco
# import serial.tools.list_ports
from pycomm3 import LogixDriver

from enum import Enum, auto



class DrawModes(Enum):
    NONE = auto()
    GRID = auto()
    SINGLE_RECT = auto()
    MULTI_RECT = auto()

class GridPrompts(Enum):
    NONE = auto()
    ROWS = auto()
    COLUMNS = auto()

drawMode = DrawModes.NONE
gridPrompt = GridPrompts.NONE
# Store time to calc FPS
pTime = 0
cTime = 0

# Window capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try indices 1, 2, or a different backend.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

win = "Detect Squares"
warp_win = "Warped"

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
# Output size
aruco_width, aruco_height = 400, 800

# PLC
PLC_IP = '172.17.35.20/2'
plc = LogixDriver(PLC_IP)
gridStartIndex = 30
multiRecStartIndex = 30
try:
    plc.open()
    print("Connected:", plc.connected)
except Exception as e:
    print(e)

# Arduino
# ports = serial.tools.list_ports.comports()
# serialInst = serial.Serial()
# portsList = []

# for port in ports:
#     portsList.append(str(port))
#     print(str(port))

# com = input("Select Com Port: ")

# for i in range(len(portsList)):
#     portName = "COM" + str(com)
#     if portsList[i].startswith(portName): # check if port exists in list
#         usedPort = portName

# serialInst.baudrate = 9600
# serialInst.port = usedPort
# serialInst.open()




rectsStart = []
rectsEnd = []
currX = 0
currY = 0
rec = False
drawing = False

def drawRectangle(action, x, y, flags, *userdata):
    global currX, currY, rec, drawing, rectsStart, rectsEnd
    if drawMode != DrawModes.NONE:
        if action == cv2.EVENT_LBUTTONDOWN:
            if drawMode == DrawModes.GRID or drawMode == DrawModes.SINGLE_RECT:
                rectsStart.clear()
                rectsEnd.clear()
                rec = False       
            rectsStart.append((x, y))
            drawing = True
            # setting current mouse position to prevent random rectangles when click
            currX = x
            currY = y
        elif action == cv2.EVENT_LBUTTONUP and drawing:
            rectsEnd.append((x, y))
            rec = True
            drawing = False
        if action == cv2.EVENT_MOUSEMOVE and drawing:
            currX = x
            currY = y

        # if action == cv2.EVENT_LBUTTONDOWN:
        #     drawing = True
        #     rec = False
        #     x1 = x
        #     y1 = y
        #     # setting current mouse position to prevent random rectangles when click
        #     currX = x
        #     currY = y
        # elif action == cv2.EVENT_LBUTTONUP:
        #     rec = True
        #     drawing = False
        #     x2 = x
        #     y2 = y


        # if action == cv2.EVENT_MOUSEMOVE and drawing:
        #     currX = x
        #     currY = y


cv2.setMouseCallback(win, drawRectangle)

# Global colour bounds (apply to all draw modes)
lowerBound = np.array([90, 20, 50])
upperBound = np.array([150, 255, 255])

detectionThreshold = 0.5

def grid(img, rows, columns, borderSize):
    coorX = (rectsStart[0][0], rectsEnd[0][0])
    coorY = (rectsStart[0][1], rectsEnd[0][1])
    xmin = min(coorX)
    xmax = max(coorX)
    ymin = min(coorY)
    ymax = max(coorY)
    gridX = (xmax - xmin) / columns
    gridY = (ymax - ymin) / rows

    # ALL COLOUR RANGES (TESTING)
    # lowerBound = np.array([0, 0, 0])
    # upperBound = np.array([255, 255, 255])



    # SHOW MASK FOR TESTING PURPOSES
    # cv2.imshow("Mask", mask)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert image from BGR to HSV
    mask = cv2.inRange(hsv, lowerBound, upperBound)

    array = np.zeros((rows, columns), dtype=bool)

    for i in range(rows):
        for j in range(columns):
            startX = xmin + j * gridX
            startY = ymin + i * gridY
            endX = startX + gridX
            endY = startY + gridY
            cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 255), borderSize)
            cell_mask = mask[int(startY):int(endY), int(startX):int(endX)]

            array[i][j] = 0

            # if the ratio of target coloured pixels is greater than threshold
            pixelRatio = np.count_nonzero(cell_mask) / cell_mask.size
            detectionPasssed = pixelRatio > detectionThreshold
            array[i][j] = detectionPasssed
            if detectionPasssed:
                cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), borderSize)

    print(array)
    # serialInst.write(array.astype(np.uint8).flatten().tobytes())
    # print(plc.get_plc_info())

    # WRITE ARRAY TO PLC
    if plc.connected:
        flat = array.astype(np.int32).flatten().tolist() # Flatten array and convert to DINT
        #data = [rows, columns] + flat
        result = plc.write(f'AMR_4.Register[{gridStartIndex}]{{ {len(flat)} }}', flat)
        print("Write result:", result)
        plc.write("CV_Grid_Rows", rows)
        plc.write("CV_Grid_Columns", columns)

def multiRec(img, borderSize):
    numOfRec = len(rectsEnd)
    if numOfRec == 0:
        return
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert image from BGR to HSV
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    detectionArray = np.zeros(numOfRec, dtype=bool)
    for i in range(numOfRec):
        cv2.rectangle(img, rectsStart[i], rectsEnd[i], (0, 0, 255), borderSize)
        cell_mask = mask[rectsStart[i][1]:rectsEnd[i][1], rectsStart[i][0]:rectsEnd[i][0]]

        # if the ratio of target coloured pixels is greater than threshold
        pixelRatio = np.count_nonzero(cell_mask) / cell_mask.size
        detectionPassed = pixelRatio > detectionThreshold
        detectionArray[i] = detectionPassed
        if detectionPassed:
            cv2.rectangle(img, rectsStart[i], rectsEnd[i], (0, 255, 0), borderSize)
    print(detectionArray)

    if plc.connected:
        result = plc.write(f'AMR_7.Register[{multiRecStartIndex}]{{ {len(detectionArray)} }}', detectionArray.astype(np.int16).tolist())
        print(result)


def borderedText(img, text, org, font, scale, text_color, border_color, text_thickness=2, border_thickness=4):
    cv2.putText(img, text, org, font, scale, border_color, border_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, text_color, text_thickness, cv2.LINE_AA)



while True:
    # Error check
    success, img = cap.read()
    if not success or img is None:
        continue

    # Homography
    warped_display = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        marker_dict = {}

        if ids is not None:
            marker_dict = {}

            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                center = pts.mean(axis=0)  # average of 4 corners
                marker_dict[int(marker_id)] = center

                cv2.circle(img, tuple(center.astype(int)), 6, (0, 255, 0), -1)
                borderedText(img, str(int(marker_id)),
                            tuple(center.astype(int) + np.array([10, -10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if all(i in marker_dict for i in [0, 1, 2, 3]):
                tl = marker_dict[0]
                tr = marker_dict[1]
                br = marker_dict[3]
                bl = marker_dict[2]

                src = np.float32([tl, tr, br, bl])
                dst = np.float32([
                    [0, 0],
                    [aruco_width, 0],
                    [aruco_width, aruco_height],
                    [0, aruco_height]
                ])

                # draw quad for debugging
                pts = src.astype(int)
                for i in range(4):
                    cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (255, 0, 0), 2)

                M = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(img, M, (aruco_width, aruco_height))
                overlay = cv2.resize(warped, (400, 800))

                h, w = overlay.shape[:2]
                x_offset = img.shape[1] - w - 20
                y_offset = 20

                img[y_offset:y_offset + h, x_offset:x_offset + w] = overlay

                cv2.rectangle(img,
                              (x_offset - 2, y_offset - 2),
                              (x_offset + w + 2, y_offset + h + 2),
                              (0, 255, 0), 2)

    # Draw rectangles ONLY on warped screen
    #if warped_display is not None:
    if rec:
        if drawMode == DrawModes.GRID and gridPrompt == GridPrompts.NONE:
            cv2.rectangle(img, rectsStart[0], rectsEnd[0], (0, 255, 0), 2)
            grid(img, gridRows, gridColumns, 2)
        elif drawMode == DrawModes.MULTI_RECT:
            multiRec(img, 2)

    if drawing and len(rectsStart) > 0:
        cv2.rectangle(img, rectsStart[-1], (currX, currY), (0, 255, 0), 2)
    

    # UI Text

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    borderedText(img, str("FPS: ") + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), (0, 0, 0))
    # Instructions
    if plc.connected:
        borderedText(img, str("PLC Connected"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    else:
        borderedText(img, str("PLC Not Connected"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    borderedText(img, str("CHOOSE START INDEX (s)"), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    borderedText(img, str("EXIT DRAWING (c)"), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    borderedText(img, str("EXIT (q)"), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), (0, 0, 0))
    borderedText(img, str("GRID (g)"), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    borderedText(img, str("MULTI-DRAW (r) | Delete (d)"), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    # Information
    borderedText(img, str(f'MODE: { drawMode }'), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
    if (drawMode == DrawModes.MULTI_RECT):
        borderedText(img, str(f"# of Rectangles: {len(rectsEnd)}"), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))
    elif (drawMode == DrawModes.GRID) and gridPrompt == GridPrompts.NONE:
        borderedText(img, str(f"Grid Size: {gridRows}x{gridColumns}"), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))

    if (gridPrompt == GridPrompts.ROWS):
        borderedText(img, str("Enter Rows (1-9)"), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))
    elif (gridPrompt == GridPrompts.COLUMNS):
        borderedText(img, str("Enter Columns (1-9)"), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))

    





    

    # User inputs
    key = cv2.waitKey(1) & 0xFF

    # Delete rectangle 
    if drawMode == DrawModes.MULTI_RECT and key == ord('d') and len(rectsEnd) > 0 and len(rectsStart) > 0:
        rectsEnd.pop()
        rectsStart.pop()

    # Grid prompt
    ch = chr(key)
    if ch.isdigit():
        inputDigit = int(ch)
        if inputDigit == 0:
            continue
        if gridPrompt == GridPrompts.ROWS:
            gridRows = inputDigit
            gridPrompt = GridPrompts.COLUMNS
        elif gridPrompt == GridPrompts.COLUMNS:
            gridColumns = inputDigit
            gridPrompt = GridPrompts.NONE

    # CHANGE DRAW MODE
    if gridPrompt == GridPrompts.NONE:
        if key == ord('c'):
            drawing = False
            rec = False
            drawMode = DrawModes.NONE
        if key == ord('g'):
            # gridRows = int(input("Rows: "))
            # gridColumns = int(input("Columns: "))
            gridPrompt = GridPrompts.ROWS
            drawMode = DrawModes.GRID
            drawing = False
            rec = False
        if key == ord('r'):
            rectsStart.clear()
            rectsEnd.clear()
            drawMode = DrawModes.MULTI_RECT
            drawing = False
            rec = False

    # --- EXIT ON 'q' ---
    if key == ord('q'):
        break


    # Display Cam
    cv2.imshow(win, img)

    # if warped_display is not None:
    #     cv2.imshow("Warped", warped_display)

cv2.destroyAllWindows()