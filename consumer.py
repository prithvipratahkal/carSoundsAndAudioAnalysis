import redis
import json
import pickle
import numpy as np
import soundfile as sf
from pyAudioAnalysis import MidTermFeatures

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)
queue_name = "audio_jobs"

# Load model
with open("carsounds-sm/motorsoundsmodel", "rb") as f:
    model = pickle.load(f)

print("🎧 Listening for jobs on Redis queue...")

while True:
    _, job = redis_client.blpop(queue_name)
    try:
        job_data = json.loads(job)
        job_id = job_data["job_id"]
        filepath = job_data["filepath"]
        filename = job_data["filename"]

        print(f"📂 Processing: {filename}")

        # Load WAV file
        y, sr = sf.read(filepath)

        # Ensure audio is mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Extract mid-term features
        features, _, _ = MidTermFeatures.mid_feature_extraction(
            y, sr,
            mt_window=1.0, mt_step=1.0,
            st_window=0.05, st_step=0.05
        )

        # Take mean across time frames
        feature_vector = np.mean(features, axis=1).reshape(1, -1)

        # Predict
        predicted_label = model.predict(feature_vector)[0]

        result = {
            "status": "completed",
            "predicted_class": predicted_label,
            "confidence": 1.0  # placeholder for now
        }

        redis_client.set(f"result:{job_id}", json.dumps(result))
        print(f"✅ {filename} → {predicted_label}")

    except Exception as e:
        print(f"❌ Error processing job: {e}")
