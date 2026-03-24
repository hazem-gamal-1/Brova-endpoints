from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders.pdf import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
import os

load_dotenv()


class CodeSuggestion(BaseModel):
    rewritten_code: str


class Feedback(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    score: float
    summary: str


class Response(BaseModel):
    content: Optional[str] = None
    code_suggestion: Optional[CodeSuggestion] = None  #
    feedback: Optional[Feedback] = None


model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.8,
)


checkpointer = InMemorySaver()


class Interview:
    def __init__(
        self,
        session_id: str,
        cv_path: str,
        job_description: str,
        interviewer_personality: str = "Friendly",
        language: str = "ar",
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
            You are the AI interviewer [Brova]. (introduce yourself as Brova - AI interviewer)

            Interview language: {"استخدم العربي  بس" if language=="ar" else "English"}

            Conduct an interview with **dynamic 1–6 questions**, including technical and behavioral questions.
            Ask **follow-up questions** when needed.

            Rules:
            1. Questions are based on CV, job description, and interviewer personality.
            2. Respond **one at a time** to candidate answers, including general speaking context in `content`.
            3. Always communicate in the selected interview language: {language}.
            4. If the candidate’s answer contains **code**, provide a `code_suggestion` object:
            - `rewritten_code`: improved version of the code
            5. Do not include `feedback` yet unless the interview is finished.
            6. At the end of the interview, provide **overall feedback** in `feedback`:
            - strengths, weaknesses, suggestions, score, summary
            7. Always return JSON matching the Response schema:
            - `content`: general speaking + next question
            - `code_suggestion`: optional, only for coding answers
            - `feedback`: optional, only at the end
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
        return res
