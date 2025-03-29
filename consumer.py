import redis
import json
import time
import os
import joblib
import numpy as np
import librosa

# Load sklearn model
model = joblib.load("carsounds-sm/motorsoundsmodel")

# Load means (saved as binary .npy format)
means = np.load("carsounds-sm/motorsoundsmodelMEANS", allow_pickle=True)

# Label map — update if different
label_map = ["fan", "gearbox", "pump", "valve"]

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # Max supported by librosa
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Ensure mfcc_mean is exactly 136-dim
    target_dim = 136
    current_dim = mfcc_mean.shape[0]

    if current_dim < target_dim:
        # Pad with zeros
        padded = np.pad(mfcc_mean, (0, target_dim - current_dim), 'constant')
    elif current_dim > target_dim:
        # Truncate
        padded = mfcc_mean[:target_dim]
    else:
        padded = mfcc_mean

    return padded - means


def predict(filepath):
    features = extract_features(filepath).reshape(1, -1)
    predicted_class = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(features))
    else:
        confidence = 1.0  # fallback if model doesn't support probabilities
    return predicted_class, round(confidence, 4)

def consume_jobs():
    print("🎧 Consumer started and waiting for jobs...")

    while True:
        job_json = r.brpop("audio_jobs", timeout=5)
        if not job_json:
            continue

        _, job_data_raw = job_json
        job_data = json.loads(job_data_raw)
        job_id = job_data["job_id"]
        filepath = job_data["filepath"]

        try:
            predicted_class, confidence = predict(filepath)
            result = {
                "status": "done",
                "predicted_class": predicted_class,
                "confidence": confidence
            }
            print(f"✅ {filepath} → {predicted_class} ({confidence})")
        except Exception as e:
            result = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ Error processing {filepath}: {e}")

        r.set(f"result:{job_id}", json.dumps(result))

        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    consume_jobs()
