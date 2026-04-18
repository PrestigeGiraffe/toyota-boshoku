import cv2
import cv2.aruco as aruco

# Setup camera
cap = cv2.VideoCapture(1)  # change index if needed

# Load dictionary + detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Print marker info
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]  # 4 corner points

            print(f"Marker ID: {marker_id}")
            print(f"Corners: {c}\n")

    # Show result
    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()