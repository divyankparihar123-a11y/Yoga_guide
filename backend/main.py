from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime
import numpy as np
import joblib

try:
    import tensorflow as tf
except ImportError:
    tf = None

app = FastAPI(
    title="ZenSense AI Backend",
    description="Backend API for Real-Time Yoga Posture Detection & Correction",
    version="2.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yoga_custom_nn.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")
nn_model = None
label_encoder = None
scaler = None

if tf is not None and os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH):
    try:
        nn_model = tf.keras.models.load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Custom Neural Network and Scaler loaded successfully")
    except Exception as e:
        print(f"⚠ Skipping NN load: {e}")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pose Baselines for all 7 poses ─────────────────────────────────
# These thresholds define the geometric constraints for a "Good" pose.
# They can be overwritten by running analyze_dataset.py which reads
# the actual video dataset.

POSE_BASELINES = {
    "tadasana": {
        "name": "Tadasana (Mountain Pose)",
        "checks": {
            "spine_vertical_threshold": 0.04,
            "leg_straight_min_angle": 165,
            "arm_to_hip_max_dist": 0.15,
            "shoulder_level_threshold": 0.03,
        },
        "description": "Stand with feet together, spine straight, arms by sides.",
    },
    "vrikshasana": {
        "name": "Vrikshasana (Tree Pose)",
        "checks": {
            "bent_leg_max_angle": 140,
            "foot_above_knee": True,
            "arms_above_shoulders": True,
            "spine_vertical_threshold": 0.06,
        },
        "description": "Stand on one leg, place other foot on inner thigh, arms up.",
    },
    "bhujangasana": {
        "name": "Bhujangasana (Cobra Pose)",
        "checks": {
            "chest_above_hips": True,
            "elbow_min_angle": 130,
            "hips_grounded_threshold": 0.15,
            "head_above_shoulders": True,
        },
        "description": "Lie face down, push chest up with palms. Keep hips grounded.",
    },
    "trikonasana": {
        "name": "Trikonasana (Triangle Pose)",
        "checks": {
            "feet_spread_min_dist": 0.3,
            "legs_straight_min_angle": 160,
            "arm_triangle_position": True,
            "shoulder_vertical_diff": 0.1,
        },
        "description": "Legs wide, reach one hand to ankle, other to sky.",
    },
    "padmasana": {
        "name": "Padmasana (Lotus Pose)",
        "checks": {
            "seated_hip_knee_threshold": 0.15,
            "knee_spread_min_dist": 0.2,
            "spine_vertical_threshold": 0.05,
            "hands_on_knees_max_dist": 0.15,
        },
        "description": "Sit cross-legged, spine erect, hands on knees.",
    },
    "balasana": {
        "name": "Balasana (Child's Pose)",
        "checks": {
            "torso_folded": True,
            "knee_max_angle": 90,
            "arms_forward_or_sides": True,
            "forehead_down": True,
        },
        "description": "Kneel and fold forward, forehead to floor.",
    },
    "parvatasana": {
        "name": "Parvatasana (Mountain Stretch / Downward V)",
        "checks": {
            "hips_highest": True,
            "arm_straight_min_angle": 155,
            "leg_straight_min_angle": 155,
            "head_between_arms": True,
        },
        "description": "Push hips up into inverted V. Arms and legs straight.",
    },
}

# Load dataset-derived baselines if available
BASELINES_FILE = os.path.join(os.path.dirname(__file__), "pose_baselines.json")
dataset_baselines = {}
if os.path.exists(BASELINES_FILE):
    with open(BASELINES_FILE, "r") as f:
        dataset_baselines = json.load(f)


# ─── Models ──────────────────────────────────────────────────────────
class PoseSession(BaseModel):
    pose_name: str
    accuracy_score: float
    duration_seconds: float
    tips: Optional[List[str]] = []

class SessionResponse(BaseModel):
    status: str
    session_id: str
    timestamp: str

class NNPoseInput(BaseModel):
    landmarks: List[float]

# In-memory session store
sessions: List[dict] = []


# ─── Routes ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ZenSense AI Backend",
        "version": "2.0.0",
        "supported_poses": list(POSE_BASELINES.keys()),
    }


@app.get("/poses")
def list_poses():
    """List all 7 supported yoga poses with their baselines."""
    return POSE_BASELINES


@app.get("/poses/{pose_id}")
def get_pose(pose_id: str):
    """Get detailed baseline config for a specific pose."""
    pose_id = pose_id.lower()
    if pose_id not in POSE_BASELINES:
        raise HTTPException(status_code=404, detail=f"Pose '{pose_id}' not found. Supported: {list(POSE_BASELINES.keys())}")

    result = POSE_BASELINES[pose_id].copy()
    # Merge dataset-derived metrics if available
    if pose_id in dataset_baselines:
        result["dataset_metrics"] = dataset_baselines[pose_id]
    return result


@app.post("/sessions", response_model=SessionResponse)
def log_session(session: PoseSession):
    """Log a completed practice session."""
    ts = datetime.utcnow().isoformat()
    entry = {
        "pose_name": session.pose_name,
        "accuracy_score": session.accuracy_score,
        "duration_seconds": session.duration_seconds,
        "tips": session.tips,
        "timestamp": ts,
    }
    sessions.append(entry)
    return SessionResponse(
        status="success",
        session_id=f"ZEN_{len(sessions):04d}",
        timestamp=ts,
    )


@app.get("/sessions")
def get_sessions():
    """Retrieve all logged sessions."""
    return sessions


def normalize_landmarks(X_raw):
    X_norm = np.copy(X_raw)
    for i in range(X_norm.shape[0]):
        row = X_norm[i]
        hip_x = (row[92] + row[96]) / 2.0
        hip_y = (row[93] + row[97]) / 2.0
        hip_z = (row[94] + row[98]) / 2.0
        for j in range(33):
            X_norm[i, j*4] -= hip_x
            X_norm[i, j*4+1] -= hip_y
            X_norm[i, j*4+2] -= hip_z
        coords = [np.sqrt(X_norm[i, j*4]**2 + X_norm[i, j*4+1]**2 + X_norm[i, j*4+2]**2) for j in range(33)]
        max_dist = max(coords)
        if max_dist > 0:
            for j in range(33):
                X_norm[i, j*4] /= max_dist
                X_norm[i, j*4+1] /= max_dist
                X_norm[i, j*4+2] /= max_dist
    return X_norm


@app.post("/predict_nn")
def predict_pose_nn(payload: NNPoseInput):
    """Predicts Yoga Pose and Quality (good/avg/poor) using Neural Network"""
    if not nn_model or not label_encoder or not scaler:
        raise HTTPException(status_code=500, detail="Neural Network components not fully loaded")
    if len(payload.landmarks) != 132:
        raise HTTPException(status_code=400, detail=f"Must provide exactly 132 landmark values, got {len(payload.landmarks)}")
        
    X = np.array([payload.landmarks])
    
    # 1. Normalize
    X_norm = normalize_landmarks(X)
    
    # 2. Scale
    X_scaled = scaler.transform(X_norm)
    
    # 3. Predict
    predictions = nn_model.predict(X_scaled, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    predicted_class = label_encoder.inverse_transform([class_idx])[0]
    
    parts = predicted_class.split("_")
    pose = parts[0]
    quality = parts[1] if len(parts) > 1 else "unknown"
    
    return {
        "status": "success",
        "predicted_class": predicted_class,
        "pose": pose,
        "quality": quality,
        "confidence": confidence
    }


@app.get("/health")
def health():
    return {"status": "ok", "poses_loaded": len(POSE_BASELINES)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
