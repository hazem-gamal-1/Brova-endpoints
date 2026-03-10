import base64
from unittest import result
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
import os
import uuid
from interview import Interview
from interview_audio_component import InterviewAudioComponent

app = FastAPI(title="AI Interview API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


interview_sessions: Dict[str, Interview] = {}
audio_component = InterviewAudioComponent()


@app.get("/")
async def root():
    return {
        "message": "AI Interview API",
        "version": "1.0.0",
        "endpoints": {
            "POST /interview/start": "Upload CV + params, return first question (text + audio base64)",
            "POST /interview/answer": "Submit audio answer; returns transcription and next question (text + audio base64)",
        },
    }


@app.post("/interview/start")
async def start_interview(
    cv: UploadFile = File(...),
    job_description: str = Form("AI Engineer"),
    interviewer_personality: str = Form("Friendly"),
):

    try:
        # validate file
        if not cv.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        session_id = f"session_{uuid.uuid4().hex}"

        TMP_DIR = "/tmp"
        cv_path = f"{TMP_DIR}/{session_id}_{cv.filename}"
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(cv_path, "wb") as f:
            content = await cv.read()
            f.write(content)

        engine = Interview(
            session_id=session_id,
            cv_path=cv_path,
            job_description=job_description,
            interviewer_personality=interviewer_personality,
        )

        interview_sessions[session_id] = {
            "cv_path": cv_path,
            "engine": engine,
            "status": "in_progress",
            "job_description": job_description,
            "interviewer_personality": interviewer_personality,
        }

        result = engine.start()
        first_resp = result["structured_response"]
        question_text = (
            first_resp.content if hasattr(first_resp, "content") else str(first_resp)
        )

        # convert to audio
        audio_bytes = audio_component.convert_text_to_speech(question_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "session_id": session_id,
            "question_text": question_text,
            "audio_base64": audio_b64,
            "full_response": result["structured_response"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# AUDIO answer endpoint
@app.post("/interview/answer")
async def submit_answer_audio(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
    code: str = Form(...),
):

    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # save uploaded audio
        TMP_DIR = "/tmp"
        audio_path = f"{TMP_DIR}/{session_id}_{audio_file.filename}"
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # convert to text
        transcription = audio_component.convert_speech_to_text(audio_path)
        transcription = transcription + f"  {code}"

        os.remove(audio_path)

        engine = interview_sessions[session_id].get("engine")
        if not engine:
            raise HTTPException(status_code=400, detail="Interview not started")

        result = engine.answer(transcription)
        answer = result["structured_response"]

        question_text = answer.content if hasattr(answer, "content") else str(answer)
        audio_bytes = audio_component.convert_text_to_speech(question_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "session_id": session_id,
            "question_text": question_text,
            "audio_base64": audio_b64,
            "full_response": result["structured_response"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
