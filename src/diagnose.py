"""
Diagnostic tool to compare live camera data with training data
"""
import torch
from classifer import Classifier
from hand_tracker import clean_coordinates_for_csv, clean_data
from camera import frame_capture
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np


mp_hands = mp.solutions.hands  # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5
)

def load_model(filepath="checkpoints/hand_gesture_model.pth"):
    device = torch.device("cpu")
    model = Classifier()
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def run():
    loaded_model, device = load_model()
    print("Model loaded successfully")
    print("\nInstructions:")
    print("  Hold up a hand gesture (1-5 fingers)")
    print("  Press '1', '2', '3', '4', or '5' to tell me what gesture you're showing")
    print("  I'll show you what I see vs what training data looks like")
    print("  Press 'q' to quit\n")
    q
    # Load training data stats
    df = pd.read_csv("train_data.csv")
    training_stats = {}
    for label in [1.0, 2.0, 3.0, 4.0, 5.0]:
        samples = df[df['label'] == label].iloc[:, :-1]
        training_stats[int(label)] = {
            'mean': samples.mean().values,
            'std': samples.std().values,
            'count': len(samples)
        }
    
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
                    pred_digit = pred + 1
                    confidence = probs[0, pred].item()
                    
                    # Show all probabilities
                    print(f"\n{'='*60}")
                    print(f"Prediction: {pred_digit} (confidence: {confidence:.1%})")
                    print("All probabilities:")
                    for i in range(5):
                        print(f"  Digit {i+1}: {probs[0, i].item():.1%}")
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return
                    
                    if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                        actual_digit = key - 48
                        print(f"\n🎯 You said this is: {actual_digit}")
                        print(f"📊 I predicted: {pred_digit} ({confidence:.1%} confidence)")
                        
                        # Compare with training data
                        coords_array = np.array(coordinates)
                        train_mean = training_stats[actual_digit]['mean']
                        train_std = training_stats[actual_digit]['std']
                        
                        # Compute z-scores (how many std deviations away)
                        z_scores = np.abs((coords_array - train_mean) / (train_std + 1e-8))
                        outliers = np.sum(z_scores > 3)  # Features more than 3 std away
                        
                        print(f"\n📈 Comparison with training data for digit {actual_digit}:")
                        print(f"  Features that are >3 std away: {outliers}/63")
                        print(f"  Mean z-score: {z_scores.mean():.2f}")
                        print(f"  Max z-score: {z_scores.max():.2f}")
                        
                        if outliers > 10:
                            print("  ⚠️  WARNING: Your hand position looks VERY different from training data!")
                            print("     Consider collecting more training data in your current setup")
                        elif outliers > 5:
                            print("  ⚠️  Caution: Some differences from training data")
                        else:
                            print("  ✅ Hand position looks similar to training data")
                        
                        # Show which features differ most
                        top_diff_idx = np.argsort(z_scores)[-3:]
                        print(f"\n  Most different features (indices): {top_diff_idx.tolist()}")
            else:
                cv2.waitKey(1)

            cv2.imshow("frame", bgr)

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    run()
