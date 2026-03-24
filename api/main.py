import base64
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
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
            "--- OR use WebSocket for a real-time connection ---",
            "WS  /interview/ws/{session_id} → full duplex interview session",
        ],
        "websocket": {
            "endpoint": "WS /interview/ws/{session_id}",
            "description": "Real-time interview over a single WebSocket connection",
            "setup_first": "POST /interview/setup to get session_id, then connect via WS",
            "client_messages": {
                "start": '{"type": "start"}  →  triggers first question',
                "answer_text": '{"type": "answer_text", "text": "..."}  →  plain text answer',
                "answer_audio": '{"type": "answer_audio", "audio_b64": "...", "code": "optional"}  →  base64 audio + optional code',
            },
            "server_messages": {
                "question": '{"type": "question", "structured_response": {...}, "audio_base64": "..."}',
                "error": '{"type": "error", "detail": "..."}',
            },
        },
        "endpoints": {
            "POST /interview/setup": {
                "description": "Initialize interview session",
                "inputs": ["cv (PDF file)", "job_description", "interviewer_personality", "language", "gender"],
                "output": ["session_id"],
            },
            "GET /interview/next_question": {
                "description": "Fetch current/next question",
                "params": ["session_id"],
                "output": ["structured_response", "audio_base64"],
            },
            "POST /interview/next_question": {
                "description": "Submit answer and get next question",
                "inputs": ["session_id", "audio_file (optional)", "code (optional)"],
                "output": ["structured_response", "audio_base64"],
            },
        },
    }


# ─── Helper ───────────────────────────────────────────────────────────────────

def _build_response_payload(result: dict, audio_component: InterviewAudioComponent) -> dict:
    """Convert an Interview engine result into a sendable payload with audio."""
    answer = result["structured_response"]

    question_text = (
        answer.content
        if hasattr(answer, "content") and answer.content
        else str(answer.feedback.summary)
    )

    audio_bytes = audio_component.convert_text_to_speech(question_text)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Serialize the structured_response pydantic model to a plain dict
    structured = answer.model_dump() if hasattr(answer, "model_dump") else answer.dict()

    return {
        "type": "question",
        "structured_response": structured,
        "audio_base64": audio_b64,
    }


# ─── REST Endpoints (unchanged) ───────────────────────────────────────────────

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
        return {"session_id": session_id}
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
        audio_bytes = audio_component.convert_text_to_speech(question_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "structured_response": result["structured_response"],
            "audio_base64": audio_b64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/next_question")
async def submit_answer_audio(
    session_id: str,
    audio_file: UploadFile = File(...),
    code: str = Form(None),
):
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        TMP_DIR = "/tmp"
        audio_path = f"{TMP_DIR}/{session_id}_{audio_file.filename}"
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        audio_component = interview_sessions[session_id].get("audio_component")
        transcription = audio_component.convert_speech_to_text(audio_path)
        transcription = transcription + f"  {code}"

        os.remove(audio_path)

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
        audio_bytes = audio_component.convert_text_to_speech(question_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "structured_response": result["structured_response"],
            "audio_base64": audio_b64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/interview/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    """
    Real-time WebSocket interview endpoint.

    The client must first create a session via POST /interview/setup, then
    connect here with the returned session_id.

    Expected client message shapes (JSON):

        {"type": "start"}
            → Starts the interview, returns first question.

        {"type": "answer_text", "text": "I would use a binary search..."}
            → Plain text answer, returns next question.

        {"type": "answer_audio", "audio_b64": "<base64>", "code": "<optional code string>"}
            → Base64-encoded audio answer + optional code snippet.
              Server transcribes audio, then returns next question.

    Server always responds with:
        {"type": "question", "structured_response": {...}, "audio_base64": "..."}
    or on error:
        {"type": "error", "detail": "..."}
    """
    await websocket.accept()

    # Validate session exists
    if session_id not in interview_sessions:
        await websocket.send_json({"type": "error", "detail": "Session not found"})
        await websocket.close(code=4004)
        return

    session = interview_sessions[session_id]
    engine: Interview = session.get("engine")
    audio_component: InterviewAudioComponent = session.get("audio_component")

    if not engine or not audio_component:
        await websocket.send_json({"type": "error", "detail": "Session not properly initialized"})
        await websocket.close(code=4000)
        return

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            msg_type = message.get("type")

            # ── Start interview ──────────────────────────────────────────────
            if msg_type == "start":
                try:
                    result = engine.start()
                    payload = _build_response_payload(result, audio_component)
                    await websocket.send_json(payload)
                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})

            # ── Plain text answer ────────────────────────────────────────────
            elif msg_type == "answer_text":
                text = message.get("text", "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "detail": "Empty answer text"})
                    continue
                try:
                    result = engine.answer(text)
                    payload = _build_response_payload(result, audio_component)
                    await websocket.send_json(payload)
                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})

            # ── Audio answer (base64) ────────────────────────────────────────
            elif msg_type == "answer_audio":
                audio_b64 = message.get("audio_b64", "")
                code = message.get("code", "")

                if not audio_b64:
                    await websocket.send_json({"type": "error", "detail": "No audio_b64 provided"})
                    continue

                try:
                    # Decode and save audio to a temp file
                    audio_bytes = base64.b64decode(audio_b64)
                    TMP_DIR = "/tmp"
                    audio_path = f"{TMP_DIR}/{session_id}_ws_audio.mp3"
                    os.makedirs(TMP_DIR, exist_ok=True)
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)

                    # STT transcription
                    transcription = audio_component.convert_speech_to_text(audio_path)
                    os.remove(audio_path)

                    # Append optional code
                    if code:
                        transcription = transcription + f"  {code}"

                    result = engine.answer(transcription)
                    payload = _build_response_payload(result, audio_component)
                    await websocket.send_json(payload)

                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})

            else:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown message type '{msg_type}'. Use: start | answer_text | answer_audio",
                })

    except WebSocketDisconnect:
        # Client disconnected — clean up if needed
        pass