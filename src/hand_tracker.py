import cv2 as cv
import mediapipe as mp
from camera import frame_capture
import csv

mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_draw = mp.solutions.drawing_utils # type: ignore

def initialize_csv():
    try:
        with open("train_data.csv", 'r'):
            pass
    except:
        with open("train_data.csv", 'a') as train_init:
            writer = csv.writer(train_init, delimiter=',')
            header=[]
            for i in range(len(mp_hands.HandLandmark)):
                header.append(mp_hands.HandLandmark(i).name + "_x")
                header.append(mp_hands.HandLandmark(i).name + "_y")
                header.append(mp_hands.HandLandmark(i).name + "_z")

            header.append("label")
            writer.writerow(header)

initialize_csv()


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
    wrist_coordinates = list_landmarks[0][1]
    coordinates = []
    for _, x_y_z in list_landmarks:
        coordinates.append(x_y_z.x - wrist_coordinates.x)
        coordinates.append(x_y_z.y - wrist_coordinates.y)
        coordinates.append(x_y_z.z - wrist_coordinates.z)
    #note: i subtract each landmark coordinate from the wrist to help the classifier with translation invariance
        
    return coordinates


def run():
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

                digit_pressed = cv.waitKey(1)

                if digit_pressed in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:

                    coordinates = clean_coordinates_for_csv(landmarks)
                    coordinates.append(digit_pressed - 48) #append the label (-48 to get num not ascii)
                    with open("train_data.csv", 'a', newline='') as train:
                        writer = csv.writer(train, delimiter=',')
                        writer.writerow(coordinates)
                    print(coordinates)

                # print(f"Hand_{hand_num}: Landmarks: {landmarks}")


        cv.imshow("frame", bgr)
        if cv.waitKey(1) == ord('q'):
            break
    

if __name__ == '__main__':
    run()