import cv2 as cv
import mediapipe as mp
from camera import frame_capture
import csv

mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_draw = mp.solutions.drawing_utils # type: ignore

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5
)

def clean_data(hand_num, hand_landmarks):
    distinct_landmarks = [(mp_hands.HandLandmark(idx).name, landmark) for idx, landmark in enumerate(hand_landmarks)] #return list of tuples
    return distinct_landmarks


def clean_coordinates_for_csv(list_landmarks):
    coordinates = []
    for _, x_y_z in list_landmarks:
        coordinates.append(x_y_z.x)
        coordinates.append(x_y_z.y)
        coordinates.append(x_y_z.z)
    return coordinates

frames = frame_capture()

for frame_dict in frames:

    rgb = frame_dict['rgb']
    bgr = frame_dict['bgr']
    
    processed_hand = hand_detector.process(rgb) #process the rgb version of the frame
    
    if processed_hand.multi_hand_landmarks: # if non zero list
        for hand_num, hand_landmarks in enumerate(processed_hand.multi_hand_landmarks): #per hand
            mp_draw.draw_landmarks(
                bgr, #draw on original bgr frame of opencv
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            landmarks = clean_data(hand_num, hand_landmarks.landmark)
            
            if cv.waitKey(1) == ord('1'):
                coordinates = clean_coordinates_for_csv(landmarks)
                with open("mappings.csv", 'a', newline='') as mappings:
                    writer = csv.writer(mappings, delimiter=',')
                    writer.writerow(coordinates)
                print(coordinates)
            # print(f"Hand_{hand_num}: Landmarks: {landmarks}")
            
    cv.imshow("frame", bgr)
    if cv.waitKey(1) == ord('q'):
        break
    

                    