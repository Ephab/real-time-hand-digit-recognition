import torch
from classifer import Classifier
from hand_tracker import clean_coordinates_for_csv, clean_data
from camera import frame_capture
import mediapipe as mp
import cv2
import time


mp_hands = mp.solutions.hands  # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5
)
device = ''

def load_model(filepath="checkpoints/hand_gesture_model.pth"):
    global device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Loading model onto: {device}")

    model = Classifier()

    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model


def run():
    loaded_model = load_model()
    print("Model loaded successfully")
    
    frames = frame_capture()

    with torch.no_grad():
        for frame_dict in frames:
            rgb = frame_dict["rgb"]
            bgr = frame_dict["bgr"]

            processed_hand = hand_detector.process(rgb)

            if processed_hand.multi_hand_landmarks:
                for hand_num, hand_landmarks in enumerate(processed_hand.multi_hand_landmarks):
                    mp_draw.draw_landmarks(bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = clean_data(hand_num, hand_landmarks.landmark)
                    coordinates = clean_coordinates_for_csv(landmarks)
                    
                    x = torch.tensor(coordinates, dtype=torch.float32).reshape(1, -1).to(device)
                    logits = loaded_model(x)
                    probs = torch.softmax(logits, dim=1)

                    pred = torch.argmax(probs, dim=1).item()
                    print([1, 2, 3, 4, 5][pred], flush=True)

            cv2.imshow("frame", bgr)

            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    run()