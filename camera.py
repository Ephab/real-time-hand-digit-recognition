import cv2 as cv
import mediapipe as mp

cam = cv.VideoCapture(0)

def frame_capture():
    if not cam.isOpened():
        raise Exception("Camera couldnt open")

    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                raise Exception("Frame couldnt be read for some reason")
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            yield {"bgr":frame, "rgb":rgb_frame}
    finally:    
        cam.release()
        cv.destroyAllWindows()