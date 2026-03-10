from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders.pdf import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv


load_dotenv()


class CodeSuggestion(BaseModel):
    tips: List[str]
    rewritten_code: str


class Feedback(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    score: float
    summary: str


class Response(BaseModel):
    question: Optional[str] = None
    code_suggestion: Optional[CodeSuggestion] = None  #
    feedback: Optional[Feedback] = None


system_prompt = """
You are the AI interviewer [Brova]. Conduct an interview with **dynamic 1–6 questions**, including technical and behavioral questions, and ask **follow-up questions** when needed.

Rules:
1. Adapt questions based on CV, job description, and interviewer personality.
2. Ask questions **one at a time**. Wait for candidate response before asking the next.
3. If the candidate’s answer contains **code**:
   - Provide a `code_suggestion` with:
     - `tips` (improvements or explanations)
     - `rewritten_code` (corrected/improved code)
4. Do not include `feedback` yet unless the interview is finished.
5. At the end of the interview, provide **overall feedback** in the `feedback` field, containing:
   - strengths, weaknesses, suggestions, score, summary
6. Always return JSON matching the `Response` schema:
   - `question`: next question or null if interview finished
   - `code_suggestion`: optional for coding answers
   - `feedback`: optional, only at the end
"""


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
checkpointer = InMemorySaver()


agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    system_prompt=system_prompt,
    response_format=ToolStrategy(Response),
)


class Interview:
    def __init__(
        self,
        session_id: str,
        cv_path: str,
        job_description: str,
        interviewer_personality: str = "Friendly",
    ):
        self.session_id = session_id
        self.cv_file_path = cv_path
        self.job_desc = job_description
        self.interviewer_personality = interviewer_personality
        self.cv = self._load_cv()

    def _load_cv(self) -> str:
        """Load and parse CV content"""
        loader = PyPDFLoader(self.cv_file_path)
        docs = loader.load()
        return " ".join([doc.page_content for doc in docs])

    def start(self):
        res = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"start the interviewe for this candidate based on the following cv \n {self.cv} ,\n  job dsecription {self.job_desc},  \n interviewer_personality:{self.interviewer_personality}",
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )["structured_response"]
        return res

    def answer(self, candidate_response: str):
        res = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": candidate_response,
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )["structured_response"]
        return res
