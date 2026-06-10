from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders.pdf import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional, Literal
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
import os

load_dotenv()


class Feedback(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    score: float
    summary: str


class Response(BaseModel):
    content: Optional[str] = None
    question_type: Literal["code", "image", "other"] = "code"
    feedback: Optional["Feedback"] = None
    current_step_index: int = 0


model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.6,
)


checkpointer = InMemorySaver()


class Interview:
    def __init__(
        self,
        session_id: str,
        cv_path: str,
        job_description: str,
        interviewer_personality: str = "Friendly",
        language: str = "arabic",
    ):
        self.session_id = session_id
        self.cv_file_path = cv_path
        self.job_desc = job_description
        self.interviewer_personality = interviewer_personality
        self.cv = self._load_cv()
        self.agent = self._prepare_agent(language)

    def _load_cv(self) -> str:
        """Load and parse CV content"""
        loader = PyPDFLoader(self.cv_file_path)
        docs = loader.load()
        return " ".join([doc.page_content for doc in docs])

    def _prepare_agent(self, language):
        system_prompt = f"""
        You are Brova, an AI interviewer. Introduce yourself as "Brova - AI interviewer" in your first interaction.

        Interview language: {"استخدم العربي المصري في الانترفيو وكمان الفيدباك بس خلي المصطلحات التقنية بالانجليزي زي ما هي" if language=="arabic" else "English"}

        Conduct a dynamic interview consisting of 1 to 4 questions (mixing technical and behavioral).
        You MUST ask at least ONE coding question and ONE drawing or sketching question during the interview. Ask follow-up questions when needed.

        Strict Operating Rules for Output Fields:

        1. Context:
        Base all questions on the provided CV, job description, and your assigned interviewer personality.

        2. Field `content`:
        Respond one turn at a time. Put your conversational response and the next question entirely inside this field. (don't include interview plan in this Field (content Field ))

        3. Field `question_type`:

        - If the question REQUIRES the user to write code, set:
        `question_type = "code"`

        - If the question REQUIRES the user to draw, sketch, design a diagram, visualize architecture, create flowcharts, UML diagrams, system diagrams, or any visual representation, set:
        `question_type = "image"`

        - For all other questions, set:
        `question_type = "other"`

        5. Field `feedback`:

        - Do NOT populate the `feedback` field during the active interview.
        - ONLY upon concluding the interview, return the full detailed `feedback` object containing:
        strengths, weaknesses, suggestions, score, and summary.
        - The `score` should be a numerical value (0-100) representing how qualified the interviewee is for the job based on their responses .

        - Generate a detailed list of tasks representing the interview plan before the START of the interview.


        if user asks to end the interview or provide feedback ends the interview and provide the feedback 
        """

        agent = create_agent(
            model=model,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            response_format=ToolStrategy(Response),
        )
        return agent

    def start(self):
        res = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"start the interviewe for this candidate based on the following cv \n {self.cv} ,\n  job dsecription {self.job_desc},  \n interviewer_personality:{self.interviewer_personality}",
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )
        print(
            res["structured_response"].todos if "structured_response" in res else None
        )
        return res

    def answer(self, candidate_response: str):
        res = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": candidate_response,
                    }
                ],
            },
            {"configurable": {"thread_id": self.session_id}},
        )
        print(
            res["structured_response"].todos if "structured_response" in res else None
        )
        return res
