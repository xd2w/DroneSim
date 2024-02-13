"""file for detecting apriltags in a video stream
"""

import cv2
import numpy as np


def main():
    stream = cv2.VideoCapture(0)
    while True:
        _, frame = stream.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        # arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict)

        # option = apriltag.DetectorOptions(families="tag36h11")
        # detector = apriltag.Detector(option)
        # detect = detector.detect

        (corners, ids, rejected) = detector.detectMarkers(img)

        if len(corners) > 0:
            p = corners[0][0].astype(int)
            p = p.reshape((-1, 1, 2))
            frame = cv2.polylines(
                frame, [p], isClosed=True, color=(0, 0, 255), thickness=10
            )

        cv2.imshow("frame", frame)

        # press Q to quit
        key = cv2.waitKey(2)
        if key == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
