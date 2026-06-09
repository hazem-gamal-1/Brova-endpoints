import base64
import logging
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
import uuid
from api.interview import Interview
from api.interview_audio_handler import InterviewAudioHandler
from api.interview_image_handler import InterviewImageHandler
import json

# ---------------------------------------------------------------------------
# Logging — Vercel captures stdout/stderr from the function runtime.
# Using basicConfig with force=True ensures the handler is always attached
# even if something else already initialised the root logger.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


app = FastAPI(title="AI Interview API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Structured-Response", "X-Todos", "X-Current-Step-Index"],
)


interview_sessions: Dict[str, Interview] = {}


# ---------------------------------------------------------------------------
# Helper — re-raises HTTPException untouched, logs & wraps everything else.
# ---------------------------------------------------------------------------
def _handle_error(route: str, exc: Exception) -> None:
    """Log the full traceback to Vercel's function logs, then raise HTTP 500."""
    if isinstance(exc, HTTPException):
        # Expected errors (404, 400…) — no traceback needed, just re-raise.
        raise exc
    logger.error(
        "Unhandled exception in %s:\n%s", route, traceback.format_exc()
    )
    raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Brova - AI Interview Platform",
        "version": "1.0.0",
        "description": "RESTful API for conducting AI-powered technical interviews with real-time feedback and audio responses",
        "base_path": "/interview",
        "workflow": {
            "step_1": "POST /interview/setup - Create interview session",
            "step_2": "GET /interview/next_question - Retrieve first question with audio",
            "step_3": "POST /interview/next_question - Submit answer and receive feedback with next question",
        },
        "endpoints": {
            "POST /interview/setup": {
                "description": "Initialize a new interview session with candidate profile and job requirements",
                "required_params": {
                    "cv": "PDF file (candidate resume)",
                    "job_description": "Target job description",
                    "interviewer_personality": "Interviewer tone (e.g., Friendly, Formal, Technical)",
                    "language": "Interview language (e.g., en, es, fr)",
                    "gender": "Interviewer voice gender (male/female)",
                },
                "response": {
                    "session_id": "Unique session identifier for interview tracking",
                },
                "status_codes": ["200", "400 - Invalid PDF", "500 - Server error"],
            },
            "GET /interview/next_question": {
                "description": "Fetch the current or next interview question with streamed audio response",
                "required_params": {
                    "session_id": "Session identifier from setup endpoint",
                },
                "response": {
                    "media_type": "audio/mpeg (streaming)",
                    "headers": ["X-Structured-Response - Question data and metadata"],
                },
                "status_codes": [
                    "200",
                    "404 - Session not found",
                    "500 - Server error",
                ],
            },
            "POST /interview/next_question": {
                "description": "Submit candidate answer and receive evaluation with next question and audio feedback",
                "required_params": {
                    "session_id": "Session identifier",
                },
                "optional_params": {
                    "answer": "Text-based answer to the question",
                    "code": "Code snippet (for technical questions)",
                    "image": "Image file (for visual/diagram questions)",
                },
                "response": {
                    "media_type": "audio/mpeg (streaming)",
                    "headers": ["X-Structured-Response - Evaluation and next question"],
                },
                "status_codes": [
                    "200",
                    "404 - Session not found",
                    "500 - Server error",
                ],
            },
        },
        "features": [
            "Multi-format answer support (text, code, images)",
            "Real-time audio streaming responses",
            "Personality-driven interviewer tone",
            "Multi-language interview support",
            "Session-based interview tracking",
        ],
    }


@app.post("/interview/setup")
async def start_interview(
    cv: UploadFile = File(...),
    job_description: str = Form("AI Engineer"),
    interviewer_personality: str = Form("Friendly"),
    language: str = Form("en"),
    gender: str = Form("female"),
):
    logger.info("POST /interview/setup — file=%s job=%s", cv.filename, job_description)
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
            "audio_component": InterviewAudioHandler(gender),
        }

        logger.info("Session created: %s", session_id)
        return {"session_id": session_id}

    except Exception as e:
        _handle_error("POST /interview/setup", e)


@app.get("/interview/next_question")
async def get_first_question(session_id: str):
    logger.info("GET /interview/next_question — session=%s", session_id)
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = interview_sessions[session_id].get("engine")
        if not engine:
            raise HTTPException(status_code=400, detail="Interview not started")

        result = engine.start()
        structured_response = result["structured_response"]

        question_text = (
            structured_response.content
            if hasattr(structured_response, "content") and structured_response.content
            else str(structured_response.feedback.summary)
        )

        audio_component = interview_sessions[session_id].get("audio_component")

        def stream_audio():
            for chunk in audio_component.stream_text_to_speech(question_text):
                yield chunk

        structured_json = json.dumps(
            structured_response.model_dump(), ensure_ascii=False
        )
        structured_b64 = base64.b64encode(structured_json.encode("utf-8")).decode("ascii")

        todos_json = json.dumps(
            getattr(structured_response, "todos", []) or [], ensure_ascii=False
        )
        todos_b64 = base64.b64encode(todos_json.encode("utf-8")).decode("ascii")
        current_step = str(getattr(structured_response, "current_step_index", 0))

        logger.info(
            "GET /interview/next_question OK — session=%s step=%s",
            session_id, current_step,
        )
        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={
                "X-Structured-Response": structured_b64,
                "X-Todos": todos_b64,
                "X-Current-Step-Index": current_step,
            },
        )

    except Exception as e:
        _handle_error("GET /interview/next_question", e)


@app.post("/interview/next_question")
async def submit_answer_audio(
    session_id: str,
    answer: str = Form(None),
    code: str = Form(None),
    image: UploadFile = File(None),
):
    logger.info(
        "POST /interview/next_question — session=%s has_answer=%s has_code=%s has_image=%s",
        session_id, bool(answer), bool(code), bool(image),
    )
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        audio_component = interview_sessions[session_id].get("audio_component")

        image_text = ""
        if image:
            TMP_DIR = "/tmp"
            image_path = f"{TMP_DIR}/{session_id}_{image.filename}"
            os.makedirs(TMP_DIR, exist_ok=True)
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            image_text = InterviewImageHandler().convert_image_to_text(image_path)

        transcription = f"{answer or ''} {code or ''} {image_text if image else ''}".strip()

        engine = interview_sessions[session_id].get("engine")
        if not engine:
            raise HTTPException(status_code=400, detail="Interview not started")

        result = engine.answer(transcription)
        structured_response = result["structured_response"]

        question_text = (
            structured_response.content
            if hasattr(structured_response, "content") and structured_response.content
            else str(structured_response.feedback.summary)
        )

        def stream_audio():
            for chunk in audio_component.stream_text_to_speech(question_text):
                yield chunk

        structured_json = json.dumps(
            structured_response.model_dump(), ensure_ascii=False
        )
        structured_b64 = base64.b64encode(structured_json.encode("utf-8")).decode("ascii")

        todos_json = json.dumps(
            getattr(structured_response, "todos", []) or [], ensure_ascii=False
        )
        todos_b64 = base64.b64encode(todos_json.encode("utf-8")).decode("ascii")
        current_step = str(getattr(structured_response, "current_step_index", 0))

        logger.info(
            "POST /interview/next_question OK — session=%s step=%s",
            session_id, current_step,
        )
        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={
                "X-Structured-Response": structured_b64,
                "X-Todos": todos_b64,
                "X-Current-Step-Index": current_step,
            },
        )

    except Exception as e:
        _handle_error("POST /interview/next_question", e)