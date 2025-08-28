# detector/inference.py 
import os, json 
import numpy as np 
import torch 
import torch.nn.functional as F 
from transformers import VideoMAEForVideoClassification 
import cv2 

DEVICE = torch.device("cpu")  
MODEL_DIR = "C:/Users/Tasne/shoplift_django/models/videomae-shoplifting"

# Load inference config
with open(os.path.join(MODEL_DIR, "inference_config.json"), "r") as f: 
    INF_CFG = json.load(f) 

NUM_FRAMES = int(INF_CFG.get("num_frames", 16)) 
RESIZE = int(INF_CFG.get("resize", 224)) 
MEAN = torch.tensor(INF_CFG.get("mean", [0.485, 0.456, 0.406])).view(3, 1, 1) 
STD  = torch.tensor(INF_CFG.get("std",  [0.229, 0.224, 0.225])).view(3, 1, 1) 

# Load model once
print("🚀 Loading model from:", MODEL_DIR)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32, local_files_only=True
).to(DEVICE) 
model.eval() 
print("✅ Model loaded successfully")


# -------- Helper functions --------
def _sample_indices(num_total: int, n: int) -> np.ndarray: 
    """Uniformly sample n frame indices from total frames."""
    if num_total <= 0: 
        return np.array([], dtype=int) 
    if num_total <= n: 
        idx = list(range(num_total)) + [num_total - 1] * (n - num_total) 
        return np.array(idx, dtype=int) 
    return np.linspace(0, num_total - 1, n, dtype=int) 


def _read_video_uniform(path: str, num_frames: int) -> np.ndarray: 
    """Read video with OpenCV and return (T,H,W,C) in RGB format.""" 
    cap = cv2.VideoCapture(path) 
    if not cap.isOpened(): 
        raise RuntimeError("Failed to open video: " + path) 

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    if total <= 0: 
        frames_bgr = [] 
        while True: 
            ok, frame = cap.read() 
            if not ok: 
                break 
            frames_bgr.append(frame) 
        cap.release() 
        total = len(frames_bgr) 
        if total == 0: 
            raise RuntimeError("No frames found in video.") 
        idxs = _sample_indices(total, num_frames) 
        frames_rgb = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in idxs] 
        return np.stack(frames_rgb, axis=0) 

    idxs = _sample_indices(total, num_frames) 
    frames_rgb = [] 
    current = 0 
    target_set = set(int(i) for i in idxs.tolist()) 
    while True: 
        ok, frame = cap.read() 
        if not ok: 
            break 
        if current in target_set: 
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        current += 1 
        if len(frames_rgb) >= num_frames: 
            break 
    cap.release() 

    if len(frames_rgb) == 0: 
        raise RuntimeError("No frames selected.") 
    while len(frames_rgb) < num_frames: 
        frames_rgb.append(frames_rgb[-1]) 
    return np.stack(frames_rgb, axis=0)  # (T,H,W,C) 


def preprocess_video(frames: np.ndarray, size: int = RESIZE) -> torch.Tensor: 
    """Convert video (T,H,W,C) -> (1,T,C,H,W), resize, and normalize.""" 
    t = torch.tensor(frames, dtype=torch.float32) / 255.0 
    t = t.permute(0, 3, 1, 2)  # (T,C,H,W) 
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False) 
    t = (t - MEAN) / STD 
    t = t.unsqueeze(0)  # (1,T,C,H,W) 
    return t 


def predict_from_video_file(video_path: str) -> dict: 
    """Run model prediction on video and return label + confidence.""" 
    print("🎬 Running prediction on video:", video_path)

    frames = _read_video_uniform(video_path, NUM_FRAMES)  # (T,H,W,C) 
    print("📸 Frames shape:", frames.shape)

    pixel_values = preprocess_video(frames)  # (1,T,C,H,W) 
    print("🧪 Preprocessed tensor shape:", pixel_values.shape)

    with torch.no_grad(): 
        outputs = model(pixel_values=pixel_values.to(DEVICE)) 
        probs = torch.softmax(outputs.logits, dim=-1)[0] 
        pred_id = int(torch.argmax(probs).item()) 

        id2label = model.config.id2label
        print("🧾 id2label from config:", id2label, type(id2label))

        # Handle id2label (dict or list)
        label = None
        if isinstance(id2label, dict):
            if str(pred_id) in id2label:
                label = id2label[str(pred_id)]
            elif pred_id in id2label:
                label = id2label[pred_id]
        elif isinstance(id2label, (list, tuple)):
            if 0 <= pred_id < len(id2label):
                label = id2label[pred_id]

        if label is None:
            label = str(pred_id)  # fallback

        score = float(probs[pred_id].item()) 

    result = {"label": label, "score": round(score*100, 4), "id": pred_id}
    print("✅ Prediction result:", result)
    return result
