from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os, uuid, json, redis
from pyAudioAnalysis import audioTrainTest as aT

app = FastAPI()
r = redis.Redis()
os.makedirs("jobs", exist_ok=True)

MODEL_DIR = "models"
MODEL_NAME = "motorsoundsmodel"
MODEL_TYPE = "gradientboosting"

@app.post("/queue")
async def queue_audio(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    temp_path = f"jobs/{job_id}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    job_data = {
        "job_id": job_id,
        "filepath": temp_path,
        "filename": file.filename
    }

    r.lpush("audio_jobs", json.dumps(job_data))
    return {"message": "Job queued", "job_id": job_id}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = r.get(f"result:{job_id}")
    if result:
        return json.loads(result)
    return {"status": "pending"}

@app.post("/predict")
async def predict_sound(file: UploadFile = File(...)):
    temp_path = f"jobs/temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    c, p, p_nam = aT.file_classification(temp_path, os.path.join(MODEL_DIR, MODEL_NAME), MODEL_TYPE)
    os.remove(temp_path)

    predicted_class_index = int(c)
    return {
        "predicted_class": p_nam[predicted_class_index],
        "confidence": round(p[predicted_class_index], 3)
    }
