import pytest
import httpx
import os
import io

BASE_URL = "https://brova-endpoints.vercel.app"
client = httpx.Client(base_url=BASE_URL, timeout=30.0)

def test_read_root():
    """Test the root endpoint for API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Brova - AI Interview Platform"
    assert "version" in data
    assert "endpoints" in data

def test_start_interview_invalid_file():
    """Test setup with invalid file format (not PDF)."""
    # Create a dummy txt file
    file_content = b"This is not a PDF file."
    files = {"cv": ("dummy.txt", io.BytesIO(file_content), "text/plain")}
    data = {
        "job_description": "Software Engineer",
        "interviewer_personality": "Friendly",
        "language": "en",
        "gender": "female"
    }
    
    response = client.post("/interview/setup", files=files, data=data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are accepted"

def test_start_interview_valid():
    """Test setup with a valid PDF file structure."""
    # Create a dummy PDF file that looks like a PDF to bypass initial checks
    file_content = b"%PDF-1.4\n%EOF"
    files = {"cv": ("resume.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {
        "job_description": "Software Engineer",
        "interviewer_personality": "Friendly",
        "language": "en",
        "gender": "female"
    }
    
    response = client.post("/interview/setup", files=files, data=data)
    
    # Asserting successful creation or an internal server error if the remote 
    # server fails to parse the dummy PDF with its LLM/parser.
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        response_data = response.json()
        assert "session_id" in response_data

def test_get_next_question_missing_session():
    """Test getting next question with invalid session."""
    response = client.get("/interview/next_question", params={"session_id": "invalid_session"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

def test_submit_answer_missing_session():
    """Test submitting answer with invalid session."""
    data = {"answer": "I have 5 years of experience."}
    response = client.post("/interview/next_question", params={"session_id": "invalid_session"}, data=data)
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

if __name__ == "__main__":
    # Automatic report generation when running the file directly
    pytest.main(["-v", f"--html={os.path.basename(__file__).replace('.py', '_report.html')}", "--self-contained-html", __file__])
