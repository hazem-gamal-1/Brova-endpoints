import base64
from re import S
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
import uuid
from api.interview import Interview
from api.interview_audio_component import InterviewAudioComponent

app = FastAPI(title="AI Interview API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


interview_sessions: Dict[str, Interview] = {}


@app.get("/")
async def root():
    return {
        "message": "AI Interview API (Brova)",
        "version": "1.0.0",
        "flow": [
            "1. POST /interview/setup → create session",
            "2. GET /interview/next_question?session_id=... → get first/next question",
            "3. POST /interview/next_question → submit answer (audio/code) and get next question",
        ],
        "endpoints": {
            "POST /interview/setup": {
                "description": "Initialize interview session",
                "inputs": [
                    "cv (PDF file)",
                    "job_description",
                    "interviewer_personality",
                    "language",
                    "gender",
                ],
                "output": ["session_id"],
            },
            "GET /interview/next_question": {
                "description": "Fetch current/next question",
                "params": ["session_id"],
                "output": ["structured_response", "audio_base64"],
            },
            "POST /interview/next_question": {
                "description": "Submit answer and get next question",
                "inputs": ["session_id", "answer (optional)", "code (optional)"],
                "output": ["structured_response", "audio_base64"],
            },
        },
    }


@app.post("/interview/setup")
async def start_interview(
    cv: UploadFile = File(...),
    job_description: str = Form("AI Engineer"),
    interviewer_personality: str = Form("Friendly"),
    language: str = Form("en"),
    gender: str = Form("female"),
):

    try:
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
            language=language,
        )

        interview_sessions[session_id] = {
            "cv_path": cv_path,
            "engine": engine,
            "status": "in_progress",
            "job_description": job_description,
            "interviewer_personality": interviewer_personality,
            "audio_component": InterviewAudioComponent(gender),
        }
        return {
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/interview/next_question")
async def get_first_question(session_id: str):
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = interview_sessions[session_id].get("engine")
        if not engine:
            raise HTTPException(status_code=400, detail="Interview not started")

        result = engine.start()
        answer = result["structured_response"]

        question_text = (
            answer.content
            if hasattr(answer, "content") and answer.content
            else str(answer.feedback.summary)
        )

        audio_component = interview_sessions[session_id].get("audio_component")

        def stream_audio():
            """Stream audio chunks as they're generated"""
            for chunk in audio_component.stream_text_to_speech(question_text):
                yield chunk

        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={"X-Structured-Response": str(result["structured_response"])},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/next_question")
async def submit_answer_audio(
    session_id: str,
    answer: str = Form(None),
    code: str = Form(None),
):

    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # convert to text
        audio_component = interview_sessions[session_id].get("audio_component")

        transcription = answer + f"  {code}"

        engine = interview_sessions[session_id].get("engine")
        if not engine:
            raise HTTPException(status_code=400, detail="Interview not started")

        result = engine.answer(transcription)
        answer = result["structured_response"]

        question_text = (
            answer.content
            if hasattr(answer, "content") and answer.content
            else str(answer.feedback.summary)
        )

        def stream_audio():
            """Stream audio chunks as they're generated"""
            for chunk in audio_component.stream_text_to_speech(question_text):
                yield chunk

        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={"X-Structured-Response": str(result["structured_response"])},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
