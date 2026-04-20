# Author: Johnson Yep
# Date: April 17, 2026
# Purpose: Computer vision color detection program used to detect and communicate rack locations to PLC in order to control AMRs at Toyota Boshoku
import threading
import cv2
import time
import numpy as np
from pycomm3 import LogixDriver
from enum import Enum, auto
import json
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"

class DrawModes(Enum):
    NONE = auto()
    GRID = auto()

class Prompts(Enum):
    NONE = auto()
    ROWS = auto()
    COLUMNS = auto()
    CHOOSING_HOMOGRAPHY_POINTS = auto()

drawMode = DrawModes.NONE
prompt = Prompts.NONE
# Store time to calc FPS
pTime = 0
cTime = 0

# Window capture
class LatestFrameCamera:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.latest = None
        self.latest_ts = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.last_retry = 0.0
        self.retry_delay = 3.0
        self.stale_after = 1.0  # seconds

        self._open_camera()
        threading.Thread(target=self._reader, daemon=True).start()

    def _open_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            self.connected = self.cap.isOpened()
            if self.connected:
                print("Camera connected")
            else:
                print("Camera open failed")
                self.cap = None
        except Exception as e:
            print(f"Camera open error: {e}")
            self.connected = False
            self.cap = None

    def _mark_disconnected(self):
        self.connected = False
        with self.lock:
            self.latest = None
            self.latest_ts = 0.0
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = None

    def _reader(self):
        while self.running:
            if self.cap is None or not self.connected:
                now = time.time()
                if now - self.last_retry >= self.retry_delay:
                    self.last_retry = now
                    self._open_camera()
                time.sleep(0.1)
                continue

            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("Camera read failed, reconnecting...")
                    self._mark_disconnected()
                    time.sleep(0.2)
                    continue

                with self.lock:
                    self.latest = frame
                    self.latest_ts = time.time()

            except Exception as e:
                print(f"Camera read error: {e}")
                self._mark_disconnected()
                time.sleep(0.2)

    def read(self):
        with self.lock:
            if (
                not self.connected
                or self.latest is None
                or (time.time() - self.latest_ts) > self.stale_after
            ):
                return False, None
            return True, self.latest.copy()

    def release(self):
        self.running = False
        self._mark_disconnected()
cap = LatestFrameCamera("rtsp://admin:Maintenance1@192.168.1.64:554/Streaming/channels/101")

win = "IMM Computer Vision"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# PLC
PLC_IP = '172.17.35.20/2'
plc = LogixDriver(PLC_IP)
writeCooldown = 24
try:
    plc.open()
    print("Connected:", plc.connected)
except Exception as e:
    print(e)

currX = 0
currY = 0
rec = False
drawing = False

def drawRectangle(action, x, y):
    global currX, currY, rec, drawing
    global currentGridStart, gridObjects
    if drawMode != DrawModes.NONE and prompt == Prompts.NONE:
        if action == cv2.EVENT_LBUTTONDOWN:   
            if drawMode == DrawModes.GRID:
                currentGridStart = (x, y)
            drawing = True
            # setting current mouse position to prevent random rectangles when click
            currX = x
            currY = y
        elif action == cv2.EVENT_LBUTTONUP and drawing:
            rec = True
            drawing = False

            if drawMode == DrawModes.GRID and currentGridStart is not None:
                gridObjects.append({
                    "start": currentGridStart,
                    "end": (x, y),
                    "rows": gridRows,
                    "cols": gridColumns
                })
                currentGridStart = None
        if action == cv2.EVENT_MOUSEMOVE and drawing:
            currX = x
            currY = y

def mouse_callback(action, x, y, flags, param):
    drawRectangle(action, x, y)
    getHomographyPoints(action, x, y)

cv2.setMouseCallback(win, mouse_callback)

u_h = 170
u_s = 255
u_v = 255
l_h = 120
l_s = 20
l_v = 50
chart_height = 300
chart_width = 360
chart = np.zeros((chart_height, chart_width, 3), dtype=np.uint8)
detectionThreshold = 0.5

for x in range(chart_width):
    hue = int((x / (chart_width - 1)) * 179)
    chart[:, x] = (hue, 255, 255)
chart = cv2.cvtColor(chart, cv2.COLOR_HSV2BGR)

for h in range(0, 180, 20):
    x = int((h / 179) * (chart.shape[1] - 1))

    cv2.line(chart, (x, chart.shape[0] - 20),
             (x, chart.shape[0]), (255, 255, 255), 1)

    cv2.putText(chart, str(h),
                (x - 12, chart.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

HSV_Controls = False
HSV_Window_Created = False
hsv_win = "HSV Controls"

Distortion_Controls = False
Distortion_Window_Created = False
dist_win = "Distortion Controls"


gridRows = 0
gridColumns = 0
reverseGridDetection = False
gridObjects = []
currentGridStart = None
saveGridValues_ONS = False
savedGridValues = []
compareToSavedGridValues = False
compareDetectionThreshold = 0
def grid(img, borderSize, mask):
    global saveGridValues_ONS, savedGridValues
    if prompt == Prompts.CHOOSING_HOMOGRAPHY_POINTS:
        return

    if not gridObjects:
        return

    for gridNum, gridObj in enumerate(gridObjects):
        start = gridObj["start"]
        end = gridObj["end"]
        rows = gridObj["rows"]
        columns = gridObj["cols"]

        xmin = min(start[0], end[0])
        xmax = max(start[0], end[0])
        ymin = min(start[1], end[1])
        ymax = max(start[1], end[1])

        gridX = (xmax - xmin) / columns
        gridY = (ymax - ymin) / rows
        array = np.zeros((rows, columns), dtype=bool)


        savedValuesFull = gridNum < len(savedGridValues)
        if saveGridValues_ONS:
            if savedValuesFull:
                savedGridValues[gridNum] = np.zeros((rows, columns), dtype=float)
            else:
                savedGridValues.append(np.zeros((rows, columns), dtype=float))

        for i in range(rows):
            for j in range(columns):
                startX = xmin + j * gridX
                startY = ymin + i * gridY
                endX = startX + gridX
                endY = startY + gridY

                cv2.rectangle(img, (int(startX), int(startY)),
                              (int(endX), int(endY)), (0, 0, 255), borderSize)

                cell_mask = mask[int(startY):int(endY), int(startX):int(endX)]

                if cell_mask.size == 0:
                    pixelRatio = 0
                else:
                    pixelRatio = np.count_nonzero(cell_mask) / cell_mask.size

                if saveGridValues_ONS:
                    savedGridValues[gridNum][i][j] = pixelRatio


                if compareToSavedGridValues and savedValuesFull:
                    base = savedGridValues[gridNum][i][j]

                    if reverseGridDetection:
                        threshold = base - compareDetectionThreshold
                        detectionPassed = pixelRatio < threshold
                    else:
                        threshold = base + compareDetectionThreshold
                        detectionPassed = pixelRatio > threshold
                else:
                    threshold = detectionThreshold

                    if reverseGridDetection:
                        detectionPassed = pixelRatio < threshold
                    else:
                        detectionPassed = pixelRatio > threshold

                borderedText(img, f"{pixelRatio*100:.2f}", (int(startX), int(startY)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), (0, 0, 0), text_thickness=3, border_thickness=6)
                if reverseGridDetection:
                    borderedText(img, f"{threshold*100:.2f} (-)", (int(startX), int(startY)-30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), (0, 0, 0), text_thickness=3, border_thickness=6)
                else:
                    borderedText(img, f"{threshold*100:.2f} (+)", (int(startX), int(startY)-30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), (0, 0, 0), text_thickness=3, border_thickness=6)

                array[i][j] = detectionPassed

                if detectionPassed:
                    cv2.rectangle(img, (int(startX), int(startY)),
                                  (int(endX), int(endY)), (0, 255, 0), borderSize + 2)
        

        if plc.connected and frameCounter % writeCooldown == 0:
            #flat = array.flatten().tolist()
            #writeArray = [False] * 96
            #writeArray[:len(flat)] = flat

            furthestAvailable = 0

            foundRack = False
            for j in range(columns):
                for i in range(rows):
                    if array[i][j]:
                        foundRack = True
                        break
                if foundRack:
                    break
                furthestAvailable +=1

            plc.write(
                #(f"CV_Grids.Grid_{gridNum+1}[0]{{32}}", writeArray[:32]),
                #(f"CV_Grids.Grid_{gridNum+1}[32]{{32}}", writeArray[32:64]),
                #(f"CV_Grids.Grid_{gridNum+1}[64]{{32}}", writeArray[64:96]),
                (f"CV_Row_Place_Spots[{gridNum+1}]", furthestAvailable)
            )

    

    saveGridValues_ONS = False


def borderedText(img, text, org, font, scale, text_color, border_color, text_thickness=2, border_thickness=5):
    cv2.putText(img, text, org, font, scale, border_color, border_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, text_color, text_thickness, cv2.LINE_AA)

def saveDetection():
    data = {
        "u_h": u_h,
        "u_s": u_s,
        "u_v": u_v,
        "l_h": l_h,
        "l_s": l_s,
        "l_v": l_v,
        
        "detectionThreshold": detectionThreshold,
        "compareDetectionThreshold": compareDetectionThreshold
    }

    with open("detection_config.json", "w") as f:
        json.dump(data, f)

    print("Data saved.")

def saveRegions():
    data = {
        "drawMode": drawMode.name,
        "gridObjects": gridObjects,
        "gridRows": gridRows,
        "gridColumns": gridColumns,
        "homographyPoints": homographyPoints,
        "f_scale": f_scale,
        "k1": k1,
        "k2": k2,
        "reverseGridDetection": reverseGridDetection,
        "homographyOn": homographyOn,

        "compareToSavedGridValues": compareToSavedGridValues,
        "compareDetectionThreshold": compareDetectionThreshold,
        "savedGridValues": [
            arr.tolist() if isinstance(arr, np.ndarray) else arr
            for arr in savedGridValues
        ]
    }

    with open("region_config.json", "w") as f:
        json.dump(data, f)

    print("Regions saved.")

def loadDetection():
    global u_h, u_s, u_v, l_h, l_s, l_v, detectionThreshold, compareDetectionThreshold

    try:
        with open("detection_config.json", "r") as f:
            data = json.load(f)

        u_h = data["u_h"]
        u_s = data["u_s"]
        u_v = data["u_v"]
        l_h = data["l_h"]
        l_s = data["l_s"]
        l_v = data["l_v"]

        detectionThreshold = data["detectionThreshold"]
        compareDetectionThreshold = data["compareDetectionThreshold"]

        print("Data loaded.")

    except FileNotFoundError:
        print("No saved detection config found.")

def loadRegions():
    global drawMode, gridObjects
    global rec, gridRows, gridColumns, homographyPoints, homographyOn
    global f_scale, k1, k2, reverseGridDetection
    global savedGridValues, compareToSavedGridValues
    global compareDetectionThreshold

    try:
        with open("region_config.json", "r") as f:
            data = json.load(f)

        drawMode = DrawModes[data["drawMode"]]
        gridObjects = data.get("gridObjects", [])
        gridRows = data["gridRows"]
        gridColumns = data["gridColumns"]
        homographyPoints = data["homographyPoints"]
        rec = True
        f_scale = data["f_scale"]
        k1 = data["k1"]
        k2 = data["k2"]
        reverseGridDetection = data["reverseGridDetection"]
        homographyOn = data["homographyOn"]

        compareToSavedGridValues = data.get("compareToSavedGridValues", False)
        compareDetectionThreshold = data.get("compareDetectionThreshold", 0)

        savedGridValues = [
            np.array(arr, dtype=float)
            for arr in data.get("savedGridValues", [])
        ]

        print("Regions loaded.")

    except FileNotFoundError:
        print("No saved region config found.")


#Homography
homographyOn = False

def doNothing(x):
    pass

loadDetection()
upperBound = np.array([u_h, u_s, u_v])
lowerBound = np.array([l_h, l_s, l_v])

frameCounter = 0

homographyPoints = []
hasPoints = False
src = None
dst = None
out_w = 0
out_h = 0
H = None

current_frame_shape = (1080, 1920, 3)

def getHomographyPoints(event, x, y):
    global homographyPoints, hasPoints, src, dst, out_w, out_h, H, prompt

    if event == cv2.EVENT_LBUTTONDOWN and len(homographyPoints) < 4 and prompt == Prompts.CHOOSING_HOMOGRAPHY_POINTS:
        homographyPoints.append([x, y])
        #print(f"Point {len(homographyPoints)}: ({x}, {y})")
        hasPoints = False

    if len(homographyPoints) == 4 and not hasPoints:
        hasPoints = True
        prompt = Prompts.NONE
        src = np.float32(homographyPoints)

        top_width = np.linalg.norm(src[1] - src[0])
        bottom_width = np.linalg.norm(src[3] - src[2])
        left_height = np.linalg.norm(src[2] - src[0])
        right_height = np.linalg.norm(src[3] - src[1])

        rect_w = int(max(top_width, bottom_width))
        rect_h = int(max(left_height, right_height))

        dst = np.float32([
            [0, 0],
            [rect_w - 1, 0],
            [0, rect_h - 1],
            [rect_w - 1, rect_h - 1]
        ])

        H_base = cv2.getPerspectiveTransform(src, dst)

        h, w = current_frame_shape[:2]

        img_corners = np.float32([
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1]
        ]).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(img_corners, H_base)

        xs = warped_corners[:, 0, 0]
        ys = warped_corners[:, 0, 1]

        min_x = int(np.floor(xs.min()))
        max_x = int(np.ceil(xs.max()))
        min_y = int(np.floor(ys.min()))
        max_y = int(np.ceil(ys.max()))

        # translation so whole warped image is visible
        T = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ], dtype=np.float32)

        H = T @ H_base

        out_w = max_x - min_x
        out_h = max_y - min_y

f_scale = 1.0
k1 = -0.35
k2 = 0.15
h_img, w_img = 2160, 3840 
map1, map2 = None, None
distortion_dirty = True
distortionOn = True

def build_distortion_maps(frame_w, frame_h):
    global K, D, map1, map2, distortion_dirty

    cx, cy = frame_w / 2, frame_h / 2
    f = frame_w * f_scale

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    D = np.array([k1, k2, 0, 0], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (frame_w, frame_h), cv2.CV_16SC2
    )
    distortion_dirty = False



def createDistortionUI():
    global Distortion_Window_Created

    cv2.namedWindow(dist_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(dist_win, 500, 250)
    cv2.moveWindow(dist_win, 500, 50)

    # f_scale: 0.50 to 2.00
    cv2.createTrackbar("F x100", dist_win, 100, 200, doNothing)

    # k1: -1.000 to +1.000
    cv2.createTrackbar("K1 x1000 +1000", dist_win, 650, 2000, doNothing)

    # k2: -1.000 to +1.000
    cv2.createTrackbar("K2 x1000 +1000", dist_win, 1150, 2000, doNothing)

    cv2.setTrackbarPos("F x100", dist_win, int(f_scale * 100))
    cv2.setTrackbarPos("K1 x1000 +1000", dist_win, int(k1 * 1000 + 1000))
    cv2.setTrackbarPos("K2 x1000 +1000", dist_win, int(k2 * 1000 + 1000))

    Distortion_Window_Created = True

def updateDistortionFromUI():
    global f_scale, k1, k2, distortion_dirty, Distortion_Window_Created

    try:
        new_f_scale = cv2.getTrackbarPos("F x100", dist_win) / 100.0
        new_k1 = (cv2.getTrackbarPos("K1 x1000 +1000", dist_win) - 1000) / 1000.0
        new_k2 = (cv2.getTrackbarPos("K2 x1000 +1000", dist_win) - 1000) / 1000.0
    except cv2.error:
        Distortion_Window_Created = False
        return

    if new_f_scale != f_scale or new_k1 != k1 or new_k2 != k2:
        f_scale = new_f_scale
        k1 = new_k1
        k2 = new_k2
        distortion_dirty = True

def window_exists(win):
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False

def create_HSV_UI():
    global HSV_Window_Created
    cv2.namedWindow(hsv_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hsv_win, 400, 350)
    cv2.moveWindow(hsv_win, 50, 50)
    cv2.createTrackbar("U-H", hsv_win, 0, 179, doNothing)
    cv2.createTrackbar("L-H", hsv_win, 0, 179, doNothing)
    cv2.createTrackbar("U-S", hsv_win, 0, 255, doNothing)
    cv2.createTrackbar("L-S", hsv_win, 0, 255, doNothing)
    cv2.createTrackbar("U-V", hsv_win, 0, 255, doNothing)
    cv2.createTrackbar("L-V", hsv_win, 0, 255, doNothing)
    cv2.createTrackbar("TH_1", hsv_win, 0, 100, doNothing)
    cv2.createTrackbar("TH_2", hsv_win, 0, 100, doNothing)
    cv2.setTrackbarPos("U-H", hsv_win, u_h)
    cv2.setTrackbarPos("U-S", hsv_win, u_s)
    cv2.setTrackbarPos("U-V", hsv_win, u_v)
    cv2.setTrackbarPos("L-H", hsv_win, l_h)
    cv2.setTrackbarPos("L-S", hsv_win, l_s)
    cv2.setTrackbarPos("L-V", hsv_win, l_v)
    cv2.setTrackbarPos("TH_1", hsv_win, int(detectionThreshold*100))
    cv2.setTrackbarPos("TH_2", hsv_win, int(compareDetectionThreshold*100))
    HSV_Window_Created = True

def update_HSV_UI():
    global upperBound, lowerBound, detectionThreshold, compareDetectionThreshold, u_h, u_s, u_v, l_h, l_s, l_v
    u_h = cv2.getTrackbarPos("U-H", hsv_win)
    u_s = cv2.getTrackbarPos("U-S", hsv_win)
    u_v = cv2.getTrackbarPos("U-V", hsv_win)
    l_h = cv2.getTrackbarPos("L-H", hsv_win)
    l_s = cv2.getTrackbarPos("L-S", hsv_win)
    l_v = cv2.getTrackbarPos("L-V", hsv_win)
    upperBound = np.array([u_h, u_s, u_v])
    lowerBound = np.array([l_h, l_s, l_v])
    detectionThreshold = float(cv2.getTrackbarPos("TH_1", hsv_win)/100)
    compareDetectionThreshold = float(cv2.getTrackbarPos("TH_2", hsv_win)/100)
    chartDisplay = chart.copy()
    l_x = int((l_h / 179) * (chart.shape[1] - 1))
    u_x = int((u_h / 179) * (chart.shape[1] - 1))
    cv2.line(chartDisplay, (l_x, 0), (l_x, chart.shape[0]), (255,255,255), 2)
    cv2.line(chartDisplay, (u_x, 0), (u_x, chart.shape[0]), (0,0,0), 2)
    cv2.imshow("HSV Chart", chartDisplay)

showUI = True

try:
    while True:
        # Error check
        success, img = cap.read()
        if not success or img is None:
            blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
            borderedText(blank, "Camera disconnected", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), (0, 0, 0))
            borderedText(blank, "Press q to quit", (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), (0, 0, 0))

            cv2.imshow(win, blank)

            fullKey = cv2.waitKeyEx(1)
            key = fullKey & 0xFF
            if key == ord('q'):
                break
            cv2.waitKey(1)
            continue
        
        frameCounter += 1

        if distortionOn:
            h_img, w_img = img.shape[:2]
            if distortion_dirty or map1 is None:
                build_distortion_maps(w_img, h_img)
            img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        
        # Homography
        current_frame_shape = img.shape
        if homographyOn and hasPoints:
            img = cv2.warpPerspective(img, H, (out_w, out_h))
        else:
            for pt in homographyPoints:
                cv2.circle(img, pt, 10, (0,255,0), -1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerBound, upperBound)

        if rec:
            if drawMode == DrawModes.GRID and prompt != Prompts.CHOOSING_HOMOGRAPHY_POINTS:
                grid(img, 2, mask)


        if drawing:
            if drawMode == DrawModes.GRID and currentGridStart is not None:
                cv2.rectangle(img, currentGridStart, (currX, currY), (0, 255, 0), 2)

        # UI Text

        if showUI:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            borderedText(img, str("FPS: ") + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), (0, 0, 0))
            # Instructions
            if plc.connected:
                borderedText(img, str(f"PLC Connected ({PLC_IP})"), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), (0, 0, 0))
            else:
                borderedText(img, str("PLC Not Connected"), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), (0, 0, 0))
            borderedText(img, str(f"ADJUST DETECTION (h)"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            borderedText(img, str("EXIT DRAWING (c)"), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            borderedText(img, str("EXIT (q)"), (10, 1400), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), (0, 0, 0))
            borderedText(img, str("GRID (g) | Delete (d)"), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            borderedText(img, str("SAVE REGIONS (s) | LOAD (l)"), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            if prompt == Prompts.CHOOSING_HOMOGRAPHY_POINTS:
                borderedText(img, str("CLICK 4 POINTS IN THIS ORDER: TL -> TR -> BL -> BR"), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 50),(0, 0, 0), 10, 12)
            else:
                borderedText(img, str(f"CHOOSE HOMOGRAPHY POINTS (p): {homographyPoints}"), (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))

            borderedText(img, str(f"HOMOGRAPHY (u): {homographyOn}"), (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            borderedText(img, str(f"ADJUST LENS DISTORTION (j)"), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            borderedText(img, str(f"HIDE/SHOW UI (`)"), (10, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))

            # Information
            borderedText(img, str(f'MODE: { drawMode }'), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
            if (drawMode == DrawModes.GRID) and prompt == Prompts.NONE:
                borderedText(img, str(f"Grid Size: {gridRows}x{gridColumns}"), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))
                borderedText(img, str(f"Reversed Detection ([): {reverseGridDetection}"), (10, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))
                borderedText(img, str("Save Current Grid Values (])"), (10, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0)) 
                borderedText(img, str(f"Compare Threshold To Saved Values (\\): {compareToSavedGridValues}"), (10, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))
                borderedText(img, str(f"Number of Grids: {len(gridObjects)}"), (10, 1250), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0))

            if (prompt == Prompts.ROWS):
                borderedText(img, str("Enter Rows (1-9)"), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0), 10, 12)
            elif (prompt == Prompts.COLUMNS):
                borderedText(img, str("Enter Columns (1-9)"), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), (0, 0, 0), 10, 12)


        # Distortion adjustion
        if Distortion_Controls and not Distortion_Window_Created:
            createDistortionUI()
        elif Distortion_Controls and Distortion_Window_Created:
            updateDistortionFromUI()
        elif not Distortion_Controls and Distortion_Window_Created:
            cv2.destroyWindow(dist_win)
            Distortion_Window_Created = False

        # Colour detection change
        if HSV_Controls and not HSV_Window_Created:  
            create_HSV_UI()
        elif HSV_Controls and HSV_Window_Created:
            if window_exists(hsv_win):
                update_HSV_UI()
                img = mask
            else:
                cv2.destroyWindow("HSV Chart")
                HSV_Window_Created = False
        elif not HSV_Controls and HSV_Window_Created:
            cv2.destroyWindow(hsv_win)
            cv2.destroyWindow("HSV Chart")
            HSV_Window_Created = False

        # User inputs
        fullKey = cv2.waitKeyEx(1)
        key = fullKey & 0xFF

        # Delete grids
        if key == ord('d'):
            if drawMode == DrawModes.GRID and gridObjects:
                gridObjects.pop()

        # Grid prompt
        ch = chr(key)
        if ch.isdigit():
            inputDigit = int(ch)
            if inputDigit == 0:
                continue
            if prompt == Prompts.ROWS:
                gridRows = inputDigit
                prompt = Prompts.COLUMNS
            elif prompt == Prompts.COLUMNS:
                gridColumns = inputDigit
                prompt = Prompts.NONE
                

        # CHANGE DRAW MODE
        if prompt == Prompts.NONE:
            if key == ord('c'):
                drawing = False
                rec = False
                drawMode = DrawModes.NONE
            if key == ord('g'):
                prompt = Prompts.ROWS
                drawMode = DrawModes.GRID

        # H to change colour detection
        if key == ord('h'):
            HSV_Controls = not HSV_Controls

        # Save and load regions
        if key == ord('s'):
            saveRegions()

        if key == ord('l'):
            loadRegions()
            build_distortion_maps(w_img, h_img)

        # Toggle homography
        if key == ord('u'):
            homographyOn = not homographyOn

        if key == ord('['):
            reverseGridDetection = not reverseGridDetection

        if key == ord(']'):
            saveGridValues_ONS = not saveGridValues_ONS

        if key == ord('\\'):
            compareToSavedGridValues = not compareToSavedGridValues

        if key == ord('p'):
            homographyPoints = []
            prompt = Prompts.CHOOSING_HOMOGRAPHY_POINTS
        
        if key == ord('j'):
            Distortion_Controls = not Distortion_Controls
        
        if key == ord('`'):
            showUI = not showUI

        # --- EXIT ON 'q' ---
        if key == ord('q'):
            break
        
        cv2.imshow(win, img)
finally:
    saveDetection()
    cap.release()
    cv2.destroyAllWindows()