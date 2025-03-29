import redis
import json
import time
import os
import joblib
import numpy as np
import librosa

# Load the trained scikit-learn audio classification model
model = joblib.load("carsounds-sm/motorsoundsmodel")

# Load the mean vector used for normalization (saved as .npy binary format)
means = np.load("carsounds-sm/motorsoundsmodelMEANS", allow_pickle=True)

# List of class labels corresponding to model output indices
label_map = ["fan", "gearbox", "pump", "valve"]

# Connect to Redis server
r = redis.Redis(host="localhost", port=6379, db=0)

# Function to extract MFCC audio features from a .wav file
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)  # Load waveform
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # Extract 128 MFCCs
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Compute mean over time

    # Ensure the feature vector has exactly 136 dimensions as expected by the model
    target_dim = 136
    current_dim = mfcc_mean.shape[0]

    if current_dim < target_dim:
        padded = np.pad(mfcc_mean, (0, target_dim - current_dim), 'constant')  # Pad with zeros
    elif current_dim > target_dim:
        padded = mfcc_mean[:target_dim]  # Truncate extra values
    else:
        padded = mfcc_mean

    return padded - means  # Normalize using pre-computed mean

# Function to predict class label and confidence for an audio file
def predict(filepath):
    features = extract_features(filepath).reshape(1, -1)  # Reshape for model input
    prediction = model.predict(features)[0]  # Predict label (usually a number like 0.0, 1.0, etc.)

    # Convert numerical label to actual class name using label_map
    try:
        label_index = int(float(prediction))
        predicted_class = label_map[label_index]
    except:
        predicted_class = str(prediction)  # fallback if prediction is already a string

    # If model supports probability estimates, fetch confidence
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(features))  # Highest class probability
    else:
        confidence = 1.0  # Default fallback confidence

    return predicted_class, round(confidence, 4)

# Main loop to consume jobs from Redis queue and process them
def consume_jobs():
    print("🎧 Consumer started and waiting for jobs...")

    while True:
        # Blocking pop from Redis list "audio_jobs"
        job_json = r.brpop("audio_jobs", timeout=5)
        if not job_json:
            continue  # No job, continue waiting

        _, job_data_raw = job_json
        job_data = json.loads(job_data_raw)
        job_id = job_data["job_id"]
        filepath = job_data["filepath"]

        try:
            # Run prediction
            predicted_class, confidence = predict(filepath)
            result = {
                "status": "done",
                "predicted_class": predicted_class,
                "confidence": confidence
            }

            # Print result to console
            print(f"✅ {filepath} → Sound: '{predicted_class}' | Confidence: {confidence * 100:.2f}%")

            # Log prediction to a CSV file (optional)
            with open("predictions_log.csv", "a") as log_file:
                log_file.write(f"{job_id},{filepath},{predicted_class},{confidence}\n")

        except Exception as e:
            # In case of any error, save the error info to Redis
            result = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ Error processing {filepath}: {e}")

        # Save result to Redis using job_id as key
        r.set(f"result:{job_id}", json.dumps(result))

        # Cleanup: remove temporary uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# Entry point to start the consumer
if __name__ == "__main__":
    consume_jobs()
