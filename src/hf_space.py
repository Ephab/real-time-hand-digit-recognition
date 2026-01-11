import gradio
import test
import mediapipe as mp
import torch
from hand_tracker import clean_coordinates_for_csv, clean_data
import cv2

hand_detector = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = "cpu"
model = test.load_model("checkpoints/hand_gesture_model.pth").to(device)

def predict_num(image):
    if image is None:
        return "No image provided"
    
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    processed_hand = hand_detector.process(image)
    
    
    if processed_hand.multi_hand_landmarks:
        for hand_landmarks in processed_hand.multi_hand_landmarks:
            landmarks = clean_data(1, hand_landmarks.landmark)
            coordinates = clean_coordinates_for_csv(landmarks)
            x = torch.tensor(coordinates, dtype=torch.float32).reshape(1, -1).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
            
            prob_dict = {i: probs.reshape(-1)[i-1].item() for i in range(1,6)}
            return [image, prob_dict, f"Prediction: Number = {[1, 2, 3, 4, 5][pred]}"]
    
    return "No hand detected"
            
app = gradio.Interface(
    fn=predict_num,
    inputs=gradio.Image(sources=['webcam'], type='numpy'),
    outputs=[gradio.Image(), gradio.Label()],
    live=True
)
app.launch()
