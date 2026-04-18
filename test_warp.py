import cv2
import cv2.aruco as aruco
import numpy as np

# Camera
cap = cv2.VideoCapture(0)

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

# Output size (adjust if you want)
W, H = 800, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        marker_dict = {}

        # store detected markers
        for i, marker_id in enumerate(ids.flatten()):
            marker_dict[int(marker_id)] = corners[i][0]

        # check if all 4 markers exist
        if all(i in marker_dict for i in [0,1,2,3]):

            # get inward-facing corners
            tl = marker_dict[0][0]  # top-left
            tr = marker_dict[1][1]  # top-right
            br = marker_dict[2][2]  # bottom-right
            bl = marker_dict[3][3]  # bottom-left

            src = np.float32([tl, tr, br, bl])

            dst = np.float32([
                [0, 0],
                [W, 0],
                [W, H],
                [0, H]
            ])

            # compute warp
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame, M, (W, H))

            # show result
            cv2.imshow("Warped", warped)

    # show original
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()