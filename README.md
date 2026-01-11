# Hand Digit Recognition (1–5)

Real-time hand digit recognition from a webcam.

This project uses:
- **MediaPipe Hands** to detect 21 3D hand landmarks per frame.
- A small **PyTorch feed-forward neural network** to classify the gesture into digits **1–5**.
- Utilities to **collect your own training data** by pressing the corresponding number key while showing the gesture.
- An optional **Gradio** UI (`hf_space.py`) for a Hugging Face Spaces-style demo.

---

## What this repo contains

- **Live data collection** → `src/hand_tracker.py`
- **Live inference (webcam)** → `src/test.py`
- **Gradio webcam demo** → `src/hf_space.py`
- **Model definition** → `src/classifer.py`
- **Camera capture generator** → `src/camera.py`
- **Training notebook** → `src/training.ipynb`
- **Training data (CSV)** → `src/train_data.csv`
- **Trained weights** → `src/checkpoints/hand_gesture_model.pth`

> Note on paths: most scripts read/write files like `train_data.csv` and `checkpoints/...` using **relative paths**.
> In practice, that means you should usually run them from inside the `src/` folder.

---

## Setup

### 1) Create a virtual environment (Conda or venv)



### 2) Install dependencies

```bash
pip install -r requirements.txt
```


### 3) macOS camera permission (note for Mac users)

The first time you run anything that opens the webcam, macOS may ask to grant camera access.
If you don’t get a prompt or the camera stays black:
- System Settings → Privacy & Security → **Camera**
- Enable access for your terminal / IDE (PyCharm, VS Code, etc.)

---

## Quick start (live prediction)

Run the real-time classifier with webcam input:

```bash
cd src
python test.py
```

What you’ll see:
- A window named `frame` with hand landmarks drawn.
- The predicted digit printed to the console.

Controls:
- Press **q** to quit.

How the prediction is produced:
1. MediaPipe detects 21 hand landmarks.
2. The landmarks are transformed into a **63-float feature vector** (21 × x/y/z).
3. The neural network outputs logits for 5 classes.
4. `softmax` → `probabilities` → `argmax` → `final digit`.

---

## Collecting training data (labeling)

To record new labeled examples into `train_data.csv`:

```bash
cd src
python hand_tracker.py
```

### How labeling works

While your hand is visible in the camera window:
- Press **1** while showing the “1” gesture
- Press **2** while showing the “2” gesture
- … up to **5**

Each time a key is pressed, it appends one **new row** to `train_data.csv`.
The row contains the **63-float feature vector** + the corresponding **label** (1–5).
Controls:
- Press **q** to quit.

### CSV format / features

When the project starts, it ensures `train_data.csv` exists and has a header.
The header includes 21 landmarks × 3 coordinates, plus a `label` column, thus 64 columns in total.

The feature vector is created in `clean_coordinates_for_csv()`:
- First landmark is the **wrist**.
- Every landmark coordinate is shifted by subtracting the wrist coordinate:

```
feature = landmark - wrist
```

This gives **translation invariance**: your hand can be anywhere in the frame and the model should still work.

---
## Training

Training is done in the notebook:

- `src/training.ipynb`

Typical workflow:

1) Collect data with `hand_tracker.py`

2) Open the notebook and run training
- It should load `train_data.csv`, split into train/val, train the model, and save weights.

3) Save the trained model weights to:
- `src/checkpoints/hand_gesture_model.pth`

4) Run live inference again:
- `python test.py`


### Tips for better accuracy

- Collect balanced data: similar number of samples for each digit (1–5).
- Collect data under different conditions:
  - lighting changes
  - different distances to the camera
  - slightly different angles

---

## Neural network model (PyTorch)

Defined in `src/classifer.py` 

Architecture:
- Input: **63**
- Hidden layers: **256 → 128**
- Output: **5** logits (for digits 1–5)

Layers (in order):
- `BatchNorm1d(63)`
- `Linear(63, 256)` + `ReLU` + `Dropout(0.3)`
- `BatchNorm1d(256)`
- `Linear(256, 128)` + `ReLU` + `Dropout(0.3)`
- `BatchNorm1d(128)`
- `Linear(128, 5)`

Output interpretation:
- The network outputs **logits**.
- In `src/test.py`, logits are converted to probabilities with `softmax`.
- The highest probability class index is mapped to digit `[1, 2, 3, 4, 5]`.

---

## Gradio / Hugging Face Space demo

There’s a Gradio app in `src/hf_space.py` that takes webcam frames and outputs the predicted digit.

Run:

```bash
cd src
python hf_space.py
```

Notes:
- The Gradio interface is configured with `live=True`.

---

## Project details / implementation notes

### Camera capture

`src/camera.py` provides `frame_capture()`, a generator that yields:

```python
{"bgr": frame, "rgb": rgb_frame}
```

- OpenCV reads BGR frames.
- MediaPipe expects RGB, so the code converts BGR → RGB.

### Device selection (CPU / CUDA / Apple Silicon MPS)

`src/test.py` chooses the best available device:
1. Apple Silicon GPU: `torch.backends.mps.is_available()` → `mps`
2. NVIDIA GPU: `torch.cuda.is_available()` → `cuda`
3. Default: `cpu`

---