import cv2
import cv2.aruco as aruco

import os
print("Saving to:", os.getcwd())

# Choose dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Marker settings
marker_size = 400  # pixels (increase for higher resolution)

# Generate markers with IDs 0–3
for marker_id in range(4):
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    filename = f"aruco_4x4_id_{marker_id}.png"
    cv2.imwrite(filename, marker_img)

    print(f"Saved {filename}")