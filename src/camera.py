import cv2 as cv

def frame_capture():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Camera couldn't open")

    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                raise Exception("Frame couldn't be read for some reason")
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            yield {"bgr":frame, "rgb":rgb_frame}
    finally:    
        cam.release()
        cv.destroyAllWindows()