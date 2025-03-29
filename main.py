from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid, os, json, redis

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only: allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup
r = redis.Redis(host="localhost", port=6379, db=0)

# Make sure job folder exists
os.makedirs("jobs", exist_ok=True)

# POST /queue — Upload file and push to Redis
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


# GET /result/{job_id} — Check for prediction result
@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = r.get(f"result:{job_id}")
    if result is None:
        return {"status": "pending"}
    return json.loads(result)
