from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders.pdf import PyPDFLoader
from pydantic import BaseModel
from typing import List
from enum import Enum
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv


load_dotenv()


class Status(str, Enum):
    starting = "starting"
    technical_question = "technical_question"
    technical_followup = "technical_followup"
    behavioral_question = "behavioral_question"
    behavioral_followup = "behavioral_followup"
    finished = "finished"


class Response(BaseModel):
    """Agent response schema."""

    content: str  # Question or feedback to show
    status: Status
    interview_finished: bool
    questions_asked: int = 0  # 0-4 progress
    strengths: List[str] = []  # Only when finished
    weaknesses: List[str] = []
    role_fit: str = ""  # Only when finished


system_prompt = (
    """
You are interviewer [Brova]. Follow EXACTLY: 1 tech Q → followup → 1 behavioral Q → followup → feedback.

ALWAYS respond as JSON matching schema above.

Update status/fields based on history. Set interview_finished=true only after both Qs + followups.
""",
)

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
        """Provide the candidate's answer to continue the interview."""
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
